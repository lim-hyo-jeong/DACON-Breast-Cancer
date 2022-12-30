import argparse
import yaml
import pandas as pd
from datetime import datetime
import settings
from utils import set_seed, get_logger
import image_preprocessing
import tabular_preprocessing
import torch_trainer
import xgb_trainer
import ensemble


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    config = yaml.load(
        open(args.config_path, 'r', encoding='UTF8'), Loader=yaml.FullLoader)

    set_seed(config['seed'])
    logger = get_logger(filename=settings.LOG /
                        f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log')

    if config['task'] == 'preprocessing':
        wsipreprocessor = image_preprocessing.WSIPreprocessor(
            settings.DATA, config['image'], logger)
        wsipreprocessor.crop_and_save('train')
        wsipreprocessor.crop_and_save('test')
        wsipreprocessor.tile_and_save('train')
        wsipreprocessor.tile_and_save('test')

        df = pd.read_csv(settings.DATA / 'train.csv')
        df_test = pd.read_csv(settings.DATA / 'test.csv')
        tabular_preprocessing.preprocess_and_save(
            df, df_test, config['tabular'], logger)

    else:
        df = pd.read_csv(settings.DATA / 'train_preprocessed.csv')
        df_test = pd.read_csv(settings.DATA / 'test_preprocessed.csv')

        if config['task'] == 'convnet_tabular_classification':
            trainer = torch_trainer.BCTrainer(config, logger)

        elif config['task'] == 'mil_classification':
            trainer = torch_trainer.BCMILTrainer(config, logger)

        elif config['task'] == 'xgb_classification':
            trainer = xgb_trainer.BCXGBTrainer(config, logger)

        if args.mode == 'train':
            trainer.train_and_validate(df)
        elif args.mode == 'inference':
            trainer.save_oof(df)
            trainer.inference(df_test)
        elif args.mode == 'ensemble':
            ensemble.stacking_models(df, config, logger)
