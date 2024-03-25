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

import flwr as fl
import gdown
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import entropy


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
from common.client import FlowerClient, get_flower_client_generator_cifar
from flwr.common import FitRes, parameters_to_ndarrays

from common.client_utils import (
    to_tensor_transform,
    get_network_generator_mlp,
    get_network_generator_cnn,
    get_model_parameters,
    get_federated_evaluation_function,
    aggregate_weighted_average,
    get_device,
    set_model_parameters,
    NetCIFAR,
    load_CIFAR_dataset
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

def fit_client_seeded(client, params, conf, seed=Seeds.DEFAULT):
    """Wrapper to always seed client training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return client.fit(params, conf)

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

np.random.seed(Seeds.DEFAULT)
random.seed(Seeds.DEFAULT)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


home_dir = content if (content := Path("/content")).exists() else Path.cwd()
dataset_dir: Path = home_dir / "cifar"
data_dir: Path = dataset_dir / "data"
centralized_partition: Path = dataset_dir / "client_data_mappings" / "centralized"
centralized_mapping: Path = dataset_dir / "client_data_mappings" / "centralized" / "0"
federated_partition: Path = dataset_dir / "client_data_mappings" / "fed_natural"

experiments_dir: Path = home_dir / "cifar" / "experiments"
weights_dir: Path = home_dir / "cifar" / "weights"
weights_dir.mkdir(exist_ok=True)
experiments_dir.mkdir(exist_ok=True)

def fit_and_save_centralised(cfg):
    torch.manual_seed(Seeds.DEFAULT)
    network_generator_cnn = get_network_generator()
    seed_net_cnn = network_generator_cnn()
    # pretrained_net = seed_net_cnn.to(DEVICE)

    centralized_flower_client_generator: Callable[
        [int], FlowerClient
    ] = get_flower_client_generator_cifar(
        model_generator=network_generator_cnn,
        partition_dir=centralized_partition,
        data_dir=data_dir,
    )

    centralized_flower_client = centralized_flower_client_generator(0)

    # Train parameters on the centralised dataset
    trained_params, num_examples, train_metrics = fit_client_seeded(
        centralized_flower_client,
        params=get_model_parameters(seed_net_cnn),
        conf=cfg,
    )

    print(train_metrics)

    torch.save(trained_params, weights_dir / 'cifar10_half_trained.pth')
    torch.save(train_metrics, weights_dir / 'cifar10_half_trained_metrics.pth')
    return

if __name__ == '__main__':
    centralized_train_config: dict[str, Any] = {
        "epochs": 2,
        "batch_size": 32,
        "client_learning_rate": 0.01,
        "weight_decay": 0.001,
        "num_workers": 2,
        "max_batches": 10000,
        "disable_tqdm": False,
    }
    fit_and_save_centralised(centralized_train_config)
