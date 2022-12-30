import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.io as io
import scipy.ndimage as ndi
import os
from tqdm import tqdm


class WSIPreprocessor:

    def __init__(self, data_path, image_params, logger):
        self.data_path = data_path
        self.image_params = image_params
        self.logger = logger

        self.df = pd.read_csv(data_path / 'train.csv')
        self.df_test = pd.read_csv(data_path / 'test.csv')

        self.df['img_path'] = str(data_path) + '/train_imgs/' + \
            self.df['img_path'].str.split('/').str[-1]
        self.df_test['img_path'] = str(data_path) + '/test_imgs/' + \
            self.df_test['img_path'].str.split('/').str[-1]

    @staticmethod
    def _read_image(image_path):
        image = io.imread(image_path)
        return image

    @staticmethod
    def _get_masked_image(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        _, a, _ = cv2.split(lab)
        th = cv2.threshold(
            a, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

        mask = np.zeros_like(a)
        mask[a < th] = 1
        mask[a >= th] = 2
        mask = ndi.binary_fill_holes(mask-1)

        masked_image = np.zeros_like(image)
        masked_image[mask == 1] = image[np.where(mask == 1)]
        masked_image[mask == 0] = 255.

        return masked_image

    @staticmethod
    def _crop_image(image):
        for w_pos in reversed(range(image.shape[1])):
            if (image[:, w_pos] == [255, 255, 255]).all():
                image = np.delete(image, w_pos, 1)
        for h_pos in reversed(range(image.shape[0])):
            if (image[h_pos, :] == [255, 255, 255]).all():
                image = np.delete(image, h_pos, 0)

        return image

    def crop_and_save(self, data_type):
        assert data_type == 'train' or data_type == 'test', "Variable 'data_type' must be 'train' or 'test'"

        if data_type == 'train':
            df = self.df
        elif data_type == 'test':
            df = self.df_test

        save_dir_name = self.image_params['crop_save_dir']
        save_dir = self.data_path / f'{data_type}_{save_dir_name}'
        os.makedirs(save_dir, exist_ok=True)

        with tqdm(total=len(df)) as pbar:
            for _, row in df.iterrows():
                file_path = save_dir / f"{row['ID']}.png"
                if os.path.exists(file_path):
                    continue

                image_path = row['img_path']
                image = self._read_image(image_path)
                masked_image = self._get_masked_image(image)
                cropped_image = self._crop_image(masked_image)
                plt.imsave(file_path, cropped_image)

                pbar.update()

        self.logger.info(f'Saved {len(df)} cropped images to {save_dir}')

    def tile_image(self, image):
        h, w, ch = image.shape

        tile_size = max(h, w) // self.image_params['tile_size_factor']
        n_tiles = self.image_params['n_tiles']

        pad_h, pad_w = (
            tile_size - h % tile_size) % tile_size, (tile_size - w % tile_size) % tile_size
        padding = [[pad_h//2, pad_h-pad_h//2],
                   [pad_w//2, pad_w-pad_w//2], [0, 0]]
        image = np.pad(image, padding, mode='constant', constant_values=255)

        image = image.reshape(
            image.shape[0]//tile_size, tile_size, image.shape[1]//tile_size, tile_size, ch)
        tiles = image.transpose(
            0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, ch)

        if len(tiles) < n_tiles:
            padding = [[0, n_tiles-len(tiles)], [0, 0], [0, 0], [0, 0]]
            tiles = np.pad(tiles, padding, mode='constant',
                           constant_values=255)

        idxs = np.argsort(tiles.reshape(tiles.shape[0], -1).sum(-1))[:n_tiles]
        tiles = tiles[idxs]

        return tiles

    def tile_and_save(self, data_type):
        assert data_type == 'train' or data_type == 'test', "Variable 'data_type' must be 'train' or 'test'"

        if data_type == 'train':
            df = self.df
        elif data_type == 'test':
            df = self.df_test

        base_img_dir_name = self.image_params['crop_save_dir']
        save_dir_name = self.image_params['tile_save_dir']
        save_dir = self.data_path / f'{data_type}_{save_dir_name}'

        os.makedirs(save_dir, exist_ok=True)

        with tqdm(total=len(df)) as pbar:
            for _, row in df.iterrows():
                if os.path.exists(save_dir / f"{row['ID']}_tile_0.png"):
                    continue

                image_path = self.data_path / \
                    f'{data_type}_{base_img_dir_name}' / f"{row['ID']}.png"
                image = self._read_image(image_path)
                tiles = self.tile_image(image)

                for idx, tile in enumerate(tiles):
                    tile_path = save_dir / f"{row['ID']}_tile_{idx}.png"
                    plt.imsave(tile_path, tile)

                pbar.update()

        self.logger.info(
            f"Saved {len(df)} images {self.image_params['n_tiles']} tiles to {save_dir}")
