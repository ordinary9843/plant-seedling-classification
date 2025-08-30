import torch
import os
from typing import Tuple, Dict
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class PlantDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        img_dir: str,
        img_size: Tuple[int, int],
        species_to_idx: Dict[str, int],
        augment: bool = False,
        is_test: bool = False,
    ):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.img_size = img_size
        self.augment = augment
        self.is_test = is_test
        self.transform = self._get_transform()
        self.species_to_idx = species_to_idx

    def _get_transform(self) -> transforms.Compose:
        transforms_list = [
            transforms.Resize(self.img_size),
        ]
        if self.augment:
            transforms_list += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5
                ),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
            ]
        transforms_list += [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        if self.augment:
            transforms_list += [
                transforms.RandomErasing(
                    p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0
                )
            ]
        return transforms.Compose(transforms_list)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.img_dir, self.dataframe.iloc[idx]["image"])
        try:
            pil_image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if self.is_test:
            label = torch.empty(1, dtype=torch.long)
        else:
            species_label = self.dataframe.iloc[idx]["species"]
            label = torch.tensor(self.species_to_idx[species_label], dtype=torch.long)
        return self.transform(pil_image), label
