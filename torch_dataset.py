import torch
from torch.utils.data import Dataset
import cv2
import settings


class BCDataset(Dataset):
    def __init__(self, ids, tabulars, labels, transforms=None):
        self.ids = ids
        self.tabulars = tabulars
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        if self.labels is not None:
            image_path = str(settings.DATA / 'train_imgs_crop' / f'{id}.png')
        else:
            image_path = str(settings.DATA / 'test_imgs_crop' / f'{id}.png')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        image = torch.as_tensor(image, dtype=torch.float)

        # tabular_feature
        tabular = self.tabulars[idx, :]
        tabular = torch.as_tensor(tabular, dtype=torch.float)

        if self.labels is not None:
            label = self.labels[idx]
            label = torch.as_tensor(label, dtype=torch.float)
            label = torch.unsqueeze(label, dim=0)
            return image, tabular, label
        else:
            return image, tabular


class BCMILDataset:
    def __init__(self, ids, n_instances, labels, transforms=None):
        self.ids = ids
        self.n_instances = n_instances
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        tiles = []

        for tile_idx in range(self.n_instances):
            if self.labels is not None:
                tile_path = str(settings.DATA / 'train_tiles' / \
                    f'{id}_tile_{tile_idx}.png')
            else:
                tile_path = str(settings.DATA / 'test_tiles' / \
                    f'{id}_tile_{tile_idx}.png')
            tile = cv2.imread(tile_path)
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            tiles.append(tile)

        assert len(tiles) == self.n_instances

        if self.transforms is not None:
            tiles = [torch.as_tensor(self.transforms(image=tile)[
                                     'image'], dtype=torch.float) for tile in tiles]
            tiles = torch.stack(tiles, dim=0)
        else:
            tiles = [torch.as_tensor(tile, dtype=torch.float)
                     for tile in tiles]
            tiles = torch.stack(tiles, dim=0)
            tiles = torch.permute(tiles, dims=(0, 3, 1, 2))

        if self.labels is not None:
            label = self.labels[idx]
            label = torch.as_tensor(label, dtype=torch.float)
            label = torch.unsqueeze(label, dim=0)
            return tiles, label
        else:
            return tiles
