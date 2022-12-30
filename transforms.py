import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def train_transforms(transform_params):
    return A.Compose([
        A.Resize(transform_params['image_size'],
                 transform_params['image_size']),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.25),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1),
            contrast_limit=(-0.1, 0.1), p=0.25
        ),
        A.Normalize(
            mean=transform_params['normalize_mean'],
            std=transform_params['normalize_std'], p=1.,
        ),
        ToTensorV2(p=1.),
    ], p=1.)


def test_transforms(transform_params):
    return A.Compose([
        A.Resize(transform_params['image_size'],
                 transform_params['image_size']),
        A.Normalize(
            mean=transform_params['normalize_mean'],
            std=transform_params['normalize_std'], p=1.,
        ),
        ToTensorV2(p=1.),
    ], p=1.)
