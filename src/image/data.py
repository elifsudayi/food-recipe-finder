"""Image data loading: transforms (with augmentation) and DataLoaders."""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import CLASSES

MEAN = (0.485, 0.456, 0.406)  # ImageNet statistics (needed later for VGG16)
STD = (0.229, 0.224, 0.225)


def build_transforms(img_size, train, augment=True):
    norm = transforms.Normalize(MEAN, STD)
    if train and augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        norm,
    ])


def get_loaders(root, img_size, batch_size=64, num_workers=2, augment=True,
                splits=("train", "val")):
    """Return ({split: DataLoader}, class_names). Test is loaded only when asked."""
    loaders, class_names = {}, None
    for split in splits:
        ds = datasets.ImageFolder(
            f"{root}/{split}",
            transform=build_transforms(img_size, train=(split == "train"), augment=augment),
        )
        if class_names is None:
            class_names = ds.classes  # ImageFolder sorts classes alphabetically
            assert set(class_names) == set(CLASSES), "class folders differ from config"
        assert ds.classes == class_names
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )
    return loaders, class_names
