import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import time
from datetime import datetime
from pytz import timezone
import settings
from metrics import macro_f1


def _get_oof(dirs):
    df_oof = pd.DataFrame(columns=dirs.keys())

    for task, dir in dirs.items():
        oof_path = settings.MODEL / dir / 'oof.npy'
        oof = np.load(oof_path)
        df_oof[task] = oof[:, 0].tolist()

    return df_oof


def _get_preds(dirs):
    df_preds = pd.DataFrame(columns=dirs.keys())

    for task, dir in dirs.items():
        preds_path = settings.MODEL / dir / 'preds.npy'
        preds = np.load(preds_path)
        df_preds[task] = preds[:, 0].tolist()

    return df_preds


def stacking_models(df, config, logger):
    start_time = time.time()
    target = config['target']
    dirs = config['dirs']

    df_sub = pd.read_csv(settings.DATA / 'sample_submission.csv')
    df_oof = _get_oof(dirs)
    df_oof[target] = df[target]
    df_preds = _get_preds(dirs)

    linear_model = LinearRegression()
    linear_model.fit(df_oof[dirs.keys()], df_oof[target])

    df_oof['stacking_result'] = np.clip(
        linear_model.predict(df_oof[dirs.keys()]), a_min=0, a_max=1)
    df_preds['stacking_result'] = np.clip(
        linear_model.predict(df_preds[dirs.keys()]), a_min=0, a_max=1)

    score = macro_f1(df_oof[target], np.where(
        df_oof['stacking_result'] > 0.5, 1, 0))

    final_preds = np.where(df_preds['stacking_result'] > 0.5, 1, 0)
    df_sub[target] = final_preds
    now = datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d-%H-%M-%S')
    submission_path = settings.SUBMISSION / f'sub-{now}.csv'
    df_sub.to_csv(submission_path, index=False)

    elapsed = time.time() - start_time
    logger.info(
        f'[Stacking ensemble] valid_macro_f1 : {score:6f} | time : {elapsed:.0f}s')
    logger.info(f'Final submission saved : {submission_path}')
