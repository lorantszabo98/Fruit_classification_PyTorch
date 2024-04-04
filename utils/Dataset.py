from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from utils import config
import os

train_transforms = transforms.Compose([
    transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
    transforms.ToTensor()
])


def get_dataloaders():

    # remove done from the directory names
    # for directory in os.listdir(config.DATA_PATH):
    #     new_name = directory.rsplit('_', 1)[0]
    #     old_path = os.path.join(config.DATA_PATH, directory)
    #     new_path = os.path.join(config.DATA_PATH, new_name)
    #     os.rename(old_path, new_path)

    full_dataset = datasets.ImageFolder(config.DATA_PATH, transform=train_transforms)

    val_test_size = int(0.3 * len(full_dataset))
    train_size = len(full_dataset) - val_test_size

    train_dataset, val_test_dataset = random_split(full_dataset, [train_size, val_test_size])

    test_size = int(0.3 * len(val_test_dataset))
    val_size = len(val_test_dataset) - test_size

    val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])

    print(f'Train size: {len(train_dataset)}')
    print(f'Val size: {len(val_dataset)}')
    print(f'Test size: {len(test_dataset)}')

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    test_dataset = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    return train_dataloader, val_dataloader, test_dataset






