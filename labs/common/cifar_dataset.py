"""Module to load the CIFAR-10 dataset."""

import csv
from pathlib import Path
from typing import Any
from collections.abc import Callable, Sequence
import torch
from PIL import Image
from PIL.Image import Image as ImageType
from torch.utils.data import Dataset
from torchvision import transforms
import io


class CIFAR(Dataset):
    """Class to load the CIFAR10 dataset."""

    def __init__(
        self,
        mapping: Path,
        data_dir: Path,
        name: str = "train",
        transform: Callable[[ImageType], Any] | None = None,
        target_transform: Callable[[int], Any] | None = None,
    ) -> None:
        """Initialize the CIFAR10 dataset.

        Args:
            mapping (Path): path to the mapping folder containing the .csv files.
            data_dir (Path): path to the dataset folder. Defaults to data_dir.
            name (str): name of the dataset to load, train or test.
            transform (Optional[Callable[[ImageType], Any]], optional):
                    transform function to be applied to the ImageType object.
            target_transform (Optional[Callable[[int], Any]], optional):
                    transform function to be applied to the label.
        """
        self.data_dir = data_dir
        self.mapping = mapping
        self.name = name

        self.data: Sequence[tuple[str, int]] = self._load_dataset()
        self.transform: Callable[[ImageType], Any] | None = transform
        self.target_transform: Callable[[int], Any] | None = target_transform

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Get a sample.

        Args:
            index (_type_): index of the sample.

        Returns
        -------
            Tuple[Any, Any]: couple (sample, label).
        """
        if 'centralized' in str(self.mapping):
          img, label = self.data[index].values()
          return img, label

        else:
          client = self.data.iloc[index]
          img = Image.open(io.BytesIO(client['sample_path'].as_py()))

          transform = transforms.Compose(
              [
                  # transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
              ]
          )
          img = transform(img)
          label = client['label']
          return img, label

        # Convert to the full path
        # print(self.data[index])
        # full_sample_path: Path = self.data_dir / self.name / sample_path

        # img: ImageType = Image.open(full_sample_path).convert("L")

        # if self.transform is not None:
        #     img = self.transform(img)

        # if self.target_transform is not None:
        #     label = self.target_transform(label)

    def __len__(self) -> int:
        """Get the length of the dataset as number of samples.

        Returns
        -------
            int: the length of the dataset.
        """
        return len(self.data)

    def _load_dataset(self) -> Sequence[tuple[str, int]]:
        """Load the paths and labels of the partition.

        Preprocess the dataset for faster future loading
        If opened for the first time

        Raises
        ------
            ValueError: raised if the mapping file doesn't exists

        Returns
        -------
            Sequence[Tuple[str, int]]:
                partition asked as a sequence of couples (path_to_file, label)
        """
        preprocessed_path: Path = (self.mapping / self.name).with_suffix(".pt")
        if preprocessed_path.exists():
            return torch.load(preprocessed_path)
        else:
            csv_path = (self.mapping / self.name).with_suffix(".csv")
            if not csv_path.exists():
                raise ValueError(f"Required files do not exist, path: {csv_path}")

            with open(csv_path) as csv_file:
                csv_reader = csv.reader(csv_file)
                # Ignore header
                next(csv_reader)

                # Extract the samples and the labels
                partition: Sequence[tuple[str, int]] = [
                    (sample_path, int(label_id))
                    for _, sample_path, _, label_id in csv_reader
                ]

                # Save for future loading
                torch.save(partition, preprocessed_path)
                return partition