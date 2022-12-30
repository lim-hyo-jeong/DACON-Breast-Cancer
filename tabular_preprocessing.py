import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import settings


def create_folds(df, n_folds, seed):
    df['fold'] = -1
    N_HG = df['N_category'].apply(str) + '_' + df['HG'].apply(str)

    skf = StratifiedKFold(
        n_splits=n_folds, random_state=seed, shuffle=True)

    for fold, (_, valid_idx) in enumerate(skf.split(df.index, N_HG)):
        df.loc[valid_idx, 'fold'] = fold

    df['fold'] = df['fold'].astype('int')

    return df


def encode(df, df_test, encode_cols):
    encoder = LabelEncoder()

    for col in encode_cols:
        df[col] = encoder.fit_transform(df[col])
        df_test[col] = encoder.transform(df_test[col])

    return df, df_test


def scaling(df, df_test, scaling_cols):
    scaler = MinMaxScaler()
    df[scaling_cols] = scaler.fit_transform(df[scaling_cols])
    df_test[scaling_cols] = scaler.transform(df_test[scaling_cols])

    return df, df_test


def replace_missing_values(df, df_test):
    # T_category
    df.loc[(df['암의 장경'] == 0) & (df['DCIS_or_LCIS_여부'] == 1), 'T_category'] = 0
    df['T_category'] = df['T_category'].fillna(
        df['암의 장경'].apply(lambda x: 1 if x <= 20 else 2 if x <= 50 else 3))
    df_test.loc[(df_test['암의 장경'] == 0) & (
        df_test['DCIS_or_LCIS_여부'] == 1), 'T_category'] = 0
    df_test['T_category'] = df_test['T_category'].fillna(
        df_test['암의 장경'].apply(lambda x: 1 if x <= 20 else 2 if x <= 50 else 3))

    # 암의 장경
    df['암의 장경'] = df['암의 장경'].fillna(df['T_category'].apply(
        lambda x: 0 if x == 0 else 13 if x == 1 else 25 if x == 2 else 60 if x == 3 else 68))
    df_test['암의 장경'] = df_test['암의 장경'].fillna(
        df_test['T_category'].apply(lambda x: 0 if x == 0 else 13 if x == 1 else 25 if x == 2 else 60 if x == 3 else 68))

    # ER
    df['ER'] = df['ER'].fillna(df['NG'].apply(
        lambda x: 1 if x in [1, 2] else 0))
    df_test['ER'] = df_test['ER'].fillna(
        df_test['NG'].apply(lambda x: 1 if x in [1, 2] else 0))

    # PR
    df['PR'] = df['PR'].fillna(df['NG'].apply(
        lambda x: 1 if x in [1, 2] else 0))
    df_test['PR'] = df['PR'].fillna(
        df_test['NG'].apply(lambda x: 1 if x in [1, 2] else 0))

    # ER_Allred_score
    df['ER_Allred_score'] = df['ER_Allred_score'].fillna(
        df['ER'].apply(lambda x: 2 if x == 0 else 7))
    df_test['ER_Allred_score'] = df_test['ER_Allred_score'].fillna(
        df_test['ER'].apply(lambda x: 2 if x == 0 else 7))

    # PR_Allred_score
    df.loc[df['PR_Allred_score'] > 8, 'PR_Allred_score'] = 8  # outlier
    df['PR_Allred_score'] = df['PR_Allred_score'].fillna(
        df['PR'].apply(lambda x: 2 if x == 0 else 6))
    df_test['PR_Allred_score'] = df_test['PR_Allred_score'].fillna(
        df_test['PR'].apply(lambda x: 2 if x == 0 else 6))

    # HER2
    df['HER2'] = df['HER2'].fillna(df['HER2_SISH'])
    df['HER2'] = df['HER2'].fillna(df['HER2_IHC'].apply(
        lambda x: 0 if x in [0, 1] else 1 if x in [2, 3] else np.NaN))
    df['HER2'] = df['HER2'].fillna(0)

    df_test['HER2'] = df_test['HER2'].fillna(df_test['HER2_SISH'])
    df_test['HER2'] = df_test['HER2'].fillna(df_test['HER2_IHC'].apply(
        lambda x: 0 if x in [0, 1] else 1 if x in [2, 3] else np.NaN))
    df_test['HER2'] = df_test['HER2'].fillna(0)

    # NG
    df['NG'] = df['NG'].fillna(df['HG_score_2'])
    df['NG'] = df['NG'].fillna(df['HG'])
    ki67_bin = df['KI-67_LI_percent'].apply(
        lambda x: 1 if x < 10 else 2 if x < 20 else 3)
    df['NG'] = df['NG'].fillna(ki67_bin.apply(
        lambda x: 2 if x in [1, 2] else 3))
    df['NG'] = df['NG'].fillna(df['HG_score_3'].apply(
        lambda x: 1 if x == 4 else 2 if x == 1 else 3))
    df['NG'] = df['NG'].fillna(df['T_category'].apply(
        lambda x: 1 if x == 0 else 2 if x in [1, 2, 3] else 3))

    df_test['NG'] = df_test['NG'].fillna(df_test['HG_score_2'])
    df_test['NG'] = df_test['NG'].fillna(df_test['HG'])
    ki67_bin = df_test['KI-67_LI_percent'].apply(
        lambda x: 1 if x < 10 else 2 if x < 20 else 3)
    df_test['NG'] = df_test['NG'].fillna(
        ki67_bin.apply(lambda x: 2 if x in [1, 2] else 3))
    df_test['NG'] = df_test['NG'].fillna(
        df_test['HG_score_3'].apply(lambda x: 1 if x == 4 else 2 if x == 1 else 3))
    df_test['NG'] = df_test['NG'].fillna(
        df_test['T_category'].apply(
            lambda x: 1 if x == 0 else 2 if x in [1, 2, 3] else 3))

    # HG, HG_score_1~3
    df['HG'] = df['HG'].fillna(df['NG'])
    df['HG_score_1'] = df['HG_score_1'].fillna(df['HG'])
    df['HG_score_2'] = df['HG_score_2'].fillna(df['NG'])
    df['HG_score_3'] = df['HG_score_3'].fillna(df['HG'])

    df_test['HG'] = df_test['HG'].fillna(df_test['NG'])
    df_test['HG_score_1'] = df_test['HG_score_1'].fillna(df_test['HG'])
    df_test['HG_score_2'] = df_test['HG_score_2'].fillna(df_test['NG'])
    df_test['HG_score_3'] = df_test['HG_score_3'].fillna(df_test['HG'])

    # KI-67_LI_percent
    df['KI-67_LI_percent'] = df['KI-67_LI_percent'].fillna(
        df['NG'].apply(lambda x: 5 if x == 1 else 10 if x == 2 else 30))
    df_test['KI-67_LI_percent'] = df_test['KI-67_LI_percent'].fillna(
        df_test['NG'].apply(lambda x: 5 if x == 1 else 10 if x == 2 else 30))

    # BRCA_mutation
    df['BRCA_mutation'] = df['BRCA_mutation'].fillna(-1)
    df_test['BRCA_mutation'] = df_test['BRCA_mutation'].fillna(-1)

    # Etc
    df = df.fillna(0)
    df_test = df_test.fillna(0)

    return df, df_test


def generate_new_features(df, df_test):
    df['수술연월일'] = pd.to_datetime(df['수술연월일'])
    df_test['수술연월일'] = pd.to_datetime(df_test['수술연월일'])
    df['수술연도'] = df['수술연월일'].dt.year
    df_test['수술연도'] = df_test['수술연월일'].dt.year

    hr = (df['ER'] == 1) | (df['PR'] == 1)
    hr_test = (df_test['ER'] == 1) | (df_test['PR'] == 1)
    df['Subtype'] = hr.astype(str) + '_' + df['HER2'].astype(str)
    df_test['Subtype'] = hr_test.astype(
        str) + '_' + df_test['HER2'].astype(str)

    return df, df_test


def preprocess_and_save(df, df_test, tabular_params, logger):
    df, df_test = replace_missing_values(df, df_test)
    df, df_test = generate_new_features(df, df_test)
    df = create_folds(df, tabular_params['n_folds'], tabular_params['seed'])
    df, df_test = encode(df, df_test, tabular_params['encode_cols'])
    df, df_test = scaling(df, df_test, tabular_params['scaling_cols'])

    df.to_csv(settings.DATA / 'train_preprocessed.csv', index=False)
    df_test.to_csv(settings.DATA / 'test_preprocessed.csv', index=False)

    logger.info(
        f"Saved preprocessed train data to {settings.DATA / 'train_preprocessed.csv'}")
    logger.info(
        f"Saved preprocessed test data to {settings.DATA / 'test_preprocessed.csv'}")

    return df, df_test
