import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from metrics import macro_f1
import settings
import pickle
import gc
import time


class BCXGBTrainer:
    def __init__(self, config, logger):
        self.config = config
        self.model_params = config['model_params']
        self.training_params = config['training_params']
        self.logger = logger

    def train_and_validate(self, df):
        self.logger.info('Run training !')
        self.logger.info(f'config : {self.config}')
        
        xgb_oof = np.zeros((df.shape[0],))
        xgb_oof_score = []
        xgb_importances = pd.DataFrame()

        model_save_dir = settings.MODEL / self.model_params['model_save_dir']
        model_save_dir.mkdir(parents=True, exist_ok=True)

        tabular_features = self.config['tabular_features']
        target = self.training_params['target']
        X = df[tabular_features]
        y = df[target]

        model = XGBClassifier(**self.training_params['best_params'])

        for fold in range(self.config['n_folds']):
            self.logger.info(f'Fold {fold} training ...')
            start_time = time.time()

            train_idx, valid_idx = df.loc[df['fold'] !=
                                          fold].index, df.loc[df['fold'] == fold].index
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            model.fit(X_train, y_train,
                      eval_set=[(X_valid, y_valid)],
                      **self.training_params['fit_params'])

            fi_tmp = pd.DataFrame()
            fi_tmp['feature'] = X_train.columns
            fi_tmp['importance'] = model.feature_importances_
            fi_tmp['fold'] = fold
            fi_tmp['seed'] = self.config['seed']
            xgb_importances = xgb_importances.append(fi_tmp)

            xgb_oof[valid_idx] = model.predict(X_valid)
            score = macro_f1(y.iloc[valid_idx], xgb_oof[valid_idx])
            xgb_oof_score.append(score)

            model_save_path = model_save_dir / f'model_f{fold}_best.pkl'
            with open(model_save_path, 'wb') as f:
                pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

            elapsed = time.time() - start_time
            self.logger.info(
                f'[Fold {fold}] valid_macro_f1 : {score:.6f} | time : {elapsed:.0f}s')
            self.logger.info(
                f"[Fold {fold}] best model saved : {model_save_path}")
            self.logger.info('-'*100)

        self.logger.info(
            f'Average best valid_macro_F1 Score: {np.mean(xgb_oof_score):.6f}')

        del model
        gc.collect()

    def inference(self, df_test):
        xgb_preds = np.zeros((df_test.shape[0], ))

        tabular_features = self.config['tabular_features']
        X_test = df_test[tabular_features]

        for fold in range(self.config['n_folds']):
            start_time = time.time()

            model_save_path = settings.MODEL / \
                self.model_params['model_save_dir'] / f'model_f{fold}_best.pkl'
            model = pickle.load(open(model_save_path, 'rb'))

            xgb_preds += model.predict_proba(X_test)[:, 1] / \
                self.config['n_folds']

            elapsed = time.time() - start_time
            self.logger.info(
                f'[model_f{fold}_best] inference time : {elapsed:.0f}s')

            del model
            gc.collect()

        xgb_preds = np.expand_dims(xgb_preds, axis=1)
        preds_save_path = settings.MODEL / \
            self.model_params['model_save_dir'] / f'preds.npy'
        np.save(preds_save_path, xgb_preds)
        self.logger.info(
            f'Prediction result saved : {preds_save_path}')

    def save_oof(self, df):
        xgb_oof = np.zeros((df.shape[0], ))
        xgb_oof_score = []

        tabular_features = self.config['tabular_features']
        target = self.training_params['target']
        X = df[tabular_features]
        y = df[target]

        for fold in range(self.config['n_folds']):
            start_time = time.time()

            model_save_path = settings.MODEL / \
                self.model_params['model_save_dir'] / f'model_f{fold}_best.pkl'
            model = pickle.load(open(model_save_path, 'rb'))

            valid_idx = df.loc[df['fold'] == fold].index
            X_valid = X.iloc[valid_idx]

            xgb_oof[valid_idx] = model.predict_proba(X_valid)[:, 1]
            score = macro_f1(y.iloc[valid_idx], np.where(
                xgb_oof[valid_idx] > 0.5, 1, 0))
            xgb_oof_score.append(score)

            elapsed = time.time() - start_time
            self.logger.info(
                f'[model_f{fold}_best] valid_macro_f1 : {score:.6f} | time : {elapsed:.0f}s')

            del model
            gc.collect()

        xgb_oof = np.expand_dims(xgb_oof, axis=1)
        oof_save_path = settings.MODEL / \
            self.model_params['model_save_dir'] / f'oof.npy'
        np.save(oof_save_path, xgb_oof)
        self.logger.info(
            f'Validation result saved : {oof_save_path}')
