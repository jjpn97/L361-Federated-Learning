# Imports

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'labs'))
import random
from collections.abc import Callable
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any
from logging import INFO
from datetime import timezone
from datetime import datetime
import shutil

import flwr as fl
import gdown
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
import pandas as pd
import torch
from torch import nn
import json
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import NDArrays, Parameters, Scalar
from flwr.common.logger import log
from flwr.server import ServerConfig, History
from flwr.server.strategy import FedAvg, Strategy, FedMedian
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from enum import IntEnum
from flwr.client import Client
from vit_pytorch import SimpleViT

import csv
from pathlib import Path
from typing import Any
from collections.abc import Callable, Sequence
import torch
import io
from PIL import Image
from PIL.Image import Image as ImageType
from torch.utils.data import Dataset

import logging
import numbers
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, cast
from collections.abc import Callable, Sized

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from flwr.common.typing import NDArrays, Scalar
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm
from flwr.common.logger import log
from sklearn.metrics import confusion_matrix
from flwr_datasets import FederatedDataset
from torchvision.transforms import Compose, ToTensor, Normalize
from flwr_datasets import FederatedDataset
from common.client import FlowerClient, get_flower_client_generator_cifar
from common.lda_utils import create_lda_partitions
from common.client_utils import (
    to_tensor_transform,
    get_network_generator_mlp,
    get_network_generator_cnn,
    get_model_parameters,
    get_federated_evaluation_function,
    aggregate_weighted_average,
    get_device,
    set_model_parameters,
    NetCIFAR
)

def get_network_generator():
    untrained_net = NetCIFAR()

    def generated_net():
        return deepcopy(untrained_net)

    return generated_net

# Add new seeds here for easy autocomplete
class Seeds(IntEnum):
    DEFAULT = 1337

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(Seeds.DEFAULT)
random.seed(Seeds.DEFAULT)
torch.manual_seed(Seeds.DEFAULT)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

home_dir = content if (content := Path("/content")).exists() else Path.cwd()
dataset_dir: Path = home_dir / "cifar"
data_dir: Path = dataset_dir / "data"
centralized_partition: Path = dataset_dir / "client_data_mappings" / "centralized"
centralized_mapping: Path = dataset_dir / "client_data_mappings" / "centralized" / "0"
federated_partition: Path = dataset_dir / "client_data_mappings" / "fed_natural"

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def load_centralised(split):
  fds = FederatedDataset(dataset="cifar10", partitioners={"train": 10})
  centralised_data = fds.load_split(split)
  centralised_data = centralised_data.with_transform(apply_transforms)
  return centralised_data

def create_partitions(concentration, total_clients, dataset_dir=''):
    # client train
    x = np.array([x[0] for x in centralized_train_dataset.data['img']])
    y = np.array([x.as_py() for x in centralized_train_dataset.data['label']])
    train_clients_partitions, dist = create_lda_partitions(
        dataset=(x, y),
        dirichlet_dist=None,
        num_partitions=total_clients,
        concentration=concentration,
        accept_imbalanced=True,
        seed=Seeds.DEFAULT,
    )

    x = np.array([x[0] for x in centralized_val_dataset.data['img']])
    y = np.array([x.as_py() for x in centralized_val_dataset.data['label']])
    val_clients_partitions, dist = create_lda_partitions(
        dataset=(x, y),
        dirichlet_dist=dist,
        num_partitions=total_clients,
        concentration=concentration,
        accept_imbalanced=True,
        seed=Seeds.DEFAULT,
    )

    x = np.array([x[0] for x in centralized_test_dataset.data['img']])
    y = np.array([x.as_py() for x in centralized_test_dataset.data['label']])
    test_clients_partitions, dist = create_lda_partitions(
        dataset=(x, y),
        dirichlet_dist=dist,
        num_partitions=total_clients,
        concentration=concentration,
        accept_imbalanced=True,
        seed=Seeds.DEFAULT,
    )

    # Store partitions
    conc_str = '01' if concentration == 0.1 else str(concentration)  

    lda_partition: Path = dataset_dir / "client_data_mappings" / f"lda_{conc_str}"
    if lda_partition.exists():
        shutil.rmtree(lda_partition)
    lda_partition.mkdir(parents=True, exist_ok=True)

    for i, (train_set, val_set, test_set) in enumerate(
        zip(train_clients_partitions, val_clients_partitions, test_clients_partitions)
    ):
        folder_path: Path = lda_partition / str(i)
        folder_path.mkdir(parents=True, exist_ok=True)

        train_path: Path = folder_path / "train.pt"
        val_path: Path = folder_path / "val.pt"
        test_path: Path = folder_path / "test.pt"

        data_train = pd.DataFrame(
            {
                "client_id": [0] * len(train_set[0]),
                "sample_path": train_set[0],
                "sample_id": range(len(train_set[0])),
                "label": train_set[1],
            }
        )
        torch.save(data_train, train_path)

        data_val = pd.DataFrame(
            {
                "client_id": [0] * len(val_set[0]),
                "sample_path": val_set[0],
                "sample_id": range(len(val_set[0])),
                "label": val_set[1],
            }
        )
        torch.save(data_val, val_path)

        data_test = pd.DataFrame(
            {
                "client_id": [0] * len(test_set[0]),
                "sample_path": test_set[0],
                "sample_id": range(len(test_set[0])),
                "label": test_set[1],
            }
        )
        torch.save(data_test, test_path)



if __name__ == '__main__':
    validation_split = 0.1
    total_clients = 100
    if not os.path.exists(centralized_mapping):
        centralized_train_dataset = load_centralised('train')
        centralized_test_dataset = load_centralised('test')
        train_valid = centralized_train_dataset.train_test_split(validation_split)
        centralized_train_dataset = train_valid["train"]
        centralized_val_dataset = train_valid["test"]
        centralized_mapping.mkdir(parents=True, exist_ok=True)
        for f in ['train.pt', 'val.pt', 'test.pt']:
            file_path = centralized_mapping / f
            if not os.path.exists(file_path):
                torch.save(centralized_train_dataset, file_path)

    # generate LDA partitions
    for concentration in [0.1, 1, 100]:
        create_partitions(concentration, total_clients, dataset_dir=dataset_dir)
