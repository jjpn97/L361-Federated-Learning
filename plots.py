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
from datetime import timezone, datetime

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

from pathlib import Path
from typing import Any
from collections.abc import Callable, Sequence
import torch
import io
from PIL import Image
from PIL.Image import Image as ImageType
from torch.utils.data import Dataset

from copy import deepcopy
from pathlib import Path
from typing import Any, cast
from collections.abc import Callable, Sized
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

import argparse

from common.client_utils import (
    get_model_parameters,
    get_federated_evaluation_function_cifar,
    aggregate_weighted_average,
    get_device,
    set_model_parameters,
    NetCIFAR,
    load_CIFAR_dataset
)

train_config: dict[str, Any] = {
    "epochs": 5,
    "batch_size": 32,
    "client_learning_rate": 0.01,
    "weight_decay": 0.001,
    "num_workers": 0,
    "max_batches": None,
    "disable_tqdm": True,
}

def _on_fit_config_fn(server_round: int) -> dict[str, Scalar]:
    return train_config | {"server_round": server_round}

test_config: dict[str, Any] = {
    "batch_size": 32,
    "num_workers": 0,
    "max_batches": None,
    "disable_tqdm": True,
}

clients_per_round = [
    [79, 68, 90, 46, 73], # Round 1
    [74, 21, 93, 99, 42], # Round 2
    [49, 81, 46, 39, 50], # Round 3
    [89, 26, 97, 84, 46], # Round 4
    [13, 54, 81, 51, 8], # Round 5
    [64, 89, 85, 44, 51], # Round 6
    [51, 94, 39, 84, 2], # Round 7
    [45, 15, 76, 58, 31], # Round 8
    [21, 40, 86, 88, 77], # Round 9
    [58, 77, 33, 25, 70], # Round 10
    [8, 81, 67, 79, 73], # Round 11
    [99, 5, 9, 52, 83], # Round 12
    [32, 33, 8, 90, 16], # Round 13
    [7, 21, 40, 10, 24], # Round 14
    [56, 69, 32, 6, 70], # Round 15
    [19, 54, 44, 77, 91], # Round 16
    [5, 55, 36, 94, 19], # Round 17
    [7, 81, 59, 44, 48], # Round 18
    [8, 26, 45, 84, 86], # Round 19
    [94, 74, 58, 3, 66], # Round 20
    [62, 69, 96, 83, 93], # Round 21
    [45, 94, 71, 72, 82], # Round 22
    [32, 33, 8, 90, 16], # Round 23
    [7, 21, 40, 10, 24], # Round 24
    [56, 55, 36, 94, 19], # Round 25
    [19, 54, 44, 77, 91], # Round 26
    [5, 55, 36, 94, 19], # Round 27
    [7, 81, 59, 44, 48], # Round 28
    [8, 26, 45, 84, 86], # Round 29
    [82, 76, 33, 91, 27] # Round 30
]


# Add new seeds here for easy autocomplete
class Seeds(IntEnum):
    DEFAULT = 1337

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(Seeds.DEFAULT)
random.seed(Seeds.DEFAULT)
torch.manual_seed(Seeds.DEFAULT)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def fit_client_seeded(client, params, conf, seed=Seeds.DEFAULT):
    """Wrapper to always seed client training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return client.fit(params, conf)

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

num_total_clients = 100
num_clients_per_round: int = 5
num_evaluate_clients: int = 0
fraction_fit: float = float(num_clients_per_round) / num_total_clients
fraction_evaluate: float = float(num_evaluate_clients) / num_total_clients

class StratWrapper:
  def __init__(self, concentration, clients_models):
    self.concentration = concentration
    self.clients_models = clients_models

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def flatten_arrays(model_params):
  weights = [x.flatten() for x in model_params] # flatten layers
  return np.concatenate(weights) # combine into 1D array

def cosine_sim(p1, p2):
  v1, v2 = flatten_arrays(p1), flatten_arrays(p2)
  return np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))

def get_network_generator():
    untrained_net = NetCIFAR()

    def generated_net():
        return deepcopy(untrained_net)

    return generated_net

federated_evaluation_function = get_federated_evaluation_function_cifar(
    data_dir=data_dir,
    centralized_mapping=centralized_mapping,
    device=get_device(),
    batch_size=test_config["batch_size"],
    num_workers=test_config["num_workers"],
    model_generator=get_network_generator(),
    criterion=nn.CrossEntropyLoss(),
)

pretrained_weights = torch.load(weights_dir / 'cifar10_half_trained.pth', map_location=torch.device(DEVICE))
network_generator_cnn = get_network_generator()
seed_net_cnn = network_generator_cnn()
pretrained_net = seed_net_cnn.to(DEVICE)
pretrained_net = set_model_parameters(pretrained_net, pretrained_weights)

class WrappedFedMed(FedMedian):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.clients_models: dict[int, list[tuple[int, NDArrays]]] = {}

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        # Call FedAvg original aggregate_fit, so that it handles the failures
        ret = super().aggregate_fit(server_round, results, failures)
        # Append clients' model parameters to the list
        self.clients_models[server_round] = [
            (i, parameters_to_ndarrays(fit_res.parameters))
            for i, (_, fit_res) in enumerate(results)
        ]
        # Return the original return value
        return ret

lda_partitions = {'0.1': dataset_dir / "client_data_mappings" / "lda_01",
                  '1':   dataset_dir / "client_data_mappings" / "lda_1",
                  '100': dataset_dir / "client_data_mappings" / "lda_100"}

# Running Fed Median experiments
def convert(o: Any) -> int | float:
    """Convert input object to Python numerical if numpy."""
    # type: ignore[reportGeneralTypeIssues]
    if isinstance(o, np.int32 | np.int64):
        return int(o)
    # type: ignore[reportGeneralTypeIssues]
    if isinstance(o, np.float32 | np.float64):
        return float(o)
    raise TypeError


def save_history(hist: History, name: str) -> None:
    """Save history from simulation to file."""
    time = int(datetime.now(timezone.utc).timestamp())
    path = home_dir / "histories"
    path.mkdir(exist_ok=True)
    path = path / f"hist_{time}_{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(hist.__dict__, f, ensure_ascii=False, indent=4, default=convert)


def start_seeded_simulation(
    client_fn: Callable[[str], Client],
    num_clients: int,
    config: ServerConfig,
    strategy: Strategy,
    name: str,
    seed: int = Seeds.DEFAULT,
    iteration: int = 0,
) -> tuple[list[tuple[int, NDArrays]], History]:
    """Wrap simulation to always seed client selection."""
    np.random.seed(seed ^ iteration)
    torch.manual_seed(seed ^ iteration)
    random.seed(seed ^ iteration)
    parameter_list, hist = fl.simulation.start_simulation_no_ray(
        client_fn=client_fn,
        num_clients=num_clients,
        client_resources={},
        config=config,
        strategy=strategy,
    )
    save_history(hist, name)
    return parameter_list, hist

federated_evaluation_function = get_federated_evaluation_function_cifar(
    data_dir=data_dir,
    centralized_mapping=centralized_mapping,
    device=get_device(),
    batch_size=test_config["batch_size"],
    num_workers=test_config["num_workers"],
    model_generator=network_generator_cnn,
    criterion=nn.CrossEntropyLoss(),
)

def _on_fit_config_fn(server_round: int) -> dict[str, Scalar]:
    return train_config | {"server_round": server_round}

def run_lda_experiments(num_rounds=10, num_clients_per_round=5, num_iterations=3, num_evaluate_clients=0):
    fraction_fit: float = float(num_clients_per_round) / num_total_clients
    fraction_evaluate: float = float(num_evaluate_clients) / num_total_clients
    for concentration in [0.1, 1, 100]:
        strat_iters = []
        lda_partition = lda_partitions[str(concentration)]
        lda_flower_client_generator = get_flower_client_generator_cifar(
            model_generator=network_generator_cnn,
            data_dir=data_dir,
            partition_dir=lda_partition,
        )
        for iter in range(num_iterations):
            initial_parameters = ndarrays_to_parameters(get_model_parameters(pretrained_net))
            strategy = WrappedFedMed(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=num_clients_per_round,
                min_evaluate_clients=num_evaluate_clients,
                min_available_clients=max(num_clients_per_round, num_evaluate_clients),
                on_fit_config_fn=_on_fit_config_fn,
                on_evaluate_config_fn=None,
                evaluate_fn=federated_evaluation_function,
                initial_parameters=initial_parameters,
                accept_failures=False,
                fit_metrics_aggregation_fn=aggregate_weighted_average,
                evaluate_metrics_aggregation_fn=aggregate_weighted_average,
            )
            strategy.concentration = concentration
            params, hist = start_seeded_simulation(
                client_fn=lambda cid: lda_flower_client_generator(cid).to_client(),
                num_clients=num_total_clients,
                config=ServerConfig(num_rounds=num_rounds),
                strategy=strategy,
                name="fedmed_lda",
                iteration=iter
            )
            strat_iters.append({'iter': iter, 'concentration': concentration, 'params': params, 'hist': hist, 'clients_models': strategy.clients_models})

        torch.save(strat_iters, weights_dir / f'strat{concentration}_iters.pt')

def plot_test_accuracy():
    accuracies = {}
    line_styles = ['-', '--', '-.']
    concentrations = [0.1, 1, 100]
    for concentration in concentrations:
        filename =  weights_dir / f'strat{concentration}_iters.pt'
        data = torch.load(filename)
        accuracies[concentration] = {}
        for item in data:
            hist = item['hist']
            for round, acc in hist.metrics_centralized['accuracy']:
                if round not in accuracies[concentration]:
                    accuracies[concentration][round] = []
                accuracies[concentration][round].append(acc)

    rounds = sorted(accuracies[concentrations[0]].keys())

    avg_accuracies = {}
    se_accuracies = {}
    print(accuracies)
    for concentration in concentrations:
        avg_accuracies[concentration] = []
        se_accuracies[concentration] = []
        for round in rounds:
            avg_acc = np.mean(accuracies[concentration][round])
            se_acc = np.std(accuracies[concentration][round]) / np.sqrt(len(accuracies[concentration][round]))
            avg_accuracies[concentration].append(avg_acc)
            se_accuracies[concentration].append(se_acc)

    plt.figure(figsize=(8, 6))
    for i, concentration in enumerate(concentrations):
        plt.plot(rounds, avg_accuracies[concentration], label=f'$\\alpha$ = {concentration}', linestyle=line_styles[i])
    plt.fill_between(rounds,
                        np.array(avg_accuracies[concentration]) - np.array(se_accuracies[concentration]),
                        np.array(avg_accuracies[concentration]) + np.array(se_accuracies[concentration]),
                        alpha=0.2)

    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Average Accuracy per Round over Iterations')
    plt.legend()
    plt.grid(True)
    plt.savefig(experiments_dir / 'test_accuracy.png', dpi=300)

def compute_class_confusion(params, network_generator, device, split='test'):
  dataset = load_CIFAR_dataset(data_dir, centralized_mapping, split)
  num_samples = len(cast(Sized, dataset))
  index_list = list(range(num_samples))
  prng = np.random.RandomState(Seeds.DEFAULT)
  prng.shuffle(index_list)
  index_list = index_list[:1500]
  dataset = torch.utils.data.Subset(dataset, index_list)

  global_net = network_generator()
  global_net = set_model_parameters(global_net, params)
  global_net = global_net.to(device)
  testloader = DataLoader(dataset=dataset, batch_size=32, num_workers=0, shuffle=False)

  correct, total = 0, 0
  global_net.eval()

  conf_matrix = torch.zeros(10, 10)
  with torch.no_grad():
      for data, labels in tqdm(testloader):
          data, labels = data.to(DEVICE), labels.to(DEVICE)
          outputs = global_net(data)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
          conf_matrix += confusion_matrix(labels.cpu(), predicted.cpu(), labels=range(10))

  conf_matrix = conf_matrix.to(torch.int)
  return conf_matrix

def get_activations_from_random_input(
    net: Module,
    device: str,
    n_samples: int = 100,
    seed: int = Seeds.DEFAULT,
) -> np.ndarray:
    """Return the activations of the network on random input."""
    # Get a random input
    prng = torch.random.manual_seed(seed)
    random_input = torch.rand((n_samples, 3, 32, 32), generator=prng)
    random_input = random_input.to(device)
    # Get the activations
    net.to(device)
    net.eval()
    with torch.no_grad():
        outputs: torch.Tensor = torch.softmax(net(random_input), dim=1)
    average_activations = torch.mean(outputs, dim=0)
    return average_activations.cpu().numpy()

def get_activations_from_centralized_test(
    net: Module,
    device: str,
    mapping,
    n_samples: int = 100,
    seed: int = Seeds.DEFAULT,
) -> np.ndarray:
    """Return the activations of the network on random input."""
    centralised_test_set = load_CIFAR_dataset(data_dir, mapping, "test").data
    # Get a random input
    prng = torch.random.manual_seed(seed)
    random_samples = torch.randint(low=0, high=len(centralised_test_set), size=(n_samples,), generator=prng)
    random_test_data = centralised_test_set[random_samples]['img']
    random_test_data = torch.stack(random_test_data).to(device)
    # Get the activations
    net.to(device)
    net.eval()
    with torch.no_grad():
        outputs: torch.Tensor = torch.softmax(net(random_test_data), dim=1)
    average_activations = torch.mean(outputs, dim=0)
    return average_activations.cpu().numpy()

# client activations per round, [NUM_ROUNDS, NUM_CLIENTS_PER_ROUND]
def compare_clients_random_activations(strat, device=DEVICE, num_rounds=30, plot=False):
  res = []
  for round in range(1, num_rounds+1):
    client_models = strat.clients_models[round]
    round_acts = []
    for _, client_params in client_models:
      client_net = set_model_parameters(network_generator_cnn(), client_params)
      act = get_activations_from_random_input(client_net, DEVICE)
      round_acts.append(act) # softmaxed activations
    res.append(round_acts)

  cos_sim = []
  kld = []
  for round in range(num_rounds):
    res_round = res[round]
    cos_sim_round = []
    kld_round = []
    for x in res_round:
      for y in res_round:
        cos_sim_round.append(cosine_sim(x, y))
        kld_round.append(entropy(x,y))
    cos_sim.append(cos_sim_round)
    kld.append(kld_round)

  if plot:
    for round in range(num_rounds):
      cos_sim_round = cos_sim[round]
      kld_round = kld[round]

      fig, axes = plt.subplots(1, 2, figsize=(6, 4))
      im1 = axes[0].imshow(np.array(cos_sim_round).reshape(num_clients_per_round, num_clients_per_round))
      im2 = axes[1].imshow(np.array(kld_round).reshape(num_clients_per_round, num_clients_per_round))
      cbar1 = fig.colorbar(im1, ax=axes[0])
      cbar2 = fig.colorbar(im2, ax=axes[1])
      fig.suptitle(f'LDA = {strat.concentration}; Round {round+1}')
      fig.tight_layout()

  return cos_sim, kld

# Activations given centralised test set
def compare_clients_test_activations(strat, device=DEVICE, num_rounds=30, plot=False):
  res = []
  for round in range(1, num_rounds+1):
    client_models = strat.clients_models[round]
    round_acts = []
    for _, client_params in client_models:
      client_net = set_model_parameters(network_generator_cnn(), client_params)
      act = get_activations_from_centralized_test(client_net, device, centralized_mapping)
      round_acts.append(act) # softmaxed activations
    res.append(round_acts)

  cos_sim = []
  kld = []
  for round in range(num_rounds):
    res_round = res[round]
    cos_sim_round = []
    kld_round = []
    for x in res_round:
      for y in res_round:
        cos_sim_round.append(cosine_sim(x, y))
        kld_round.append(entropy(x,y))
    cos_sim.append(cos_sim_round)
    kld.append(kld_round)

  # (1) compare client activations against each other
  if plot:
    for round in range(num_rounds):
      if round % 5 == 0:
        cos_sim_round = cos_sim[round]
        kld_round = kld[round]

        fig, axes = plt.subplots(1, 2, figsize=(6, 4))
        im1 = axes[0].imshow(np.array(cos_sim_round).reshape(num_clients_per_round, num_clients_per_round))
        im2 = axes[1].imshow(np.array(kld_round).reshape(num_clients_per_round, num_clients_per_round))
        cbar1 = fig.colorbar(im1, ax=axes[0])
        cbar2 = fig.colorbar(im2, ax=axes[1])
        fig.suptitle(f'Round {round+1}')
        fig.tight_layout()

  return cos_sim, kld

def clients_models_activation_similarity(strategies, num_rounds=30):
    strategy_names = ["0.1", "1", "100"]
    colors = ["r", "g", "b"]  # Colors for each strategy

    for k, func in enumerate([compare_clients_random_activations, compare_clients_test_activations]):
        fig, ax = plt.subplots(figsize=(8, 6))
        for strategy, strategy_name, color in zip(strategies, strategy_names, colors):
            cos_sim, kld = func(strategy, DEVICE, num_rounds=num_rounds)
            mean_pairwise_metrics_per_round = []
            for round in range(num_rounds):
                cos_sim_round = cos_sim[round]
                cos_sim2 = np.array(cos_sim_round).reshape((num_clients_per_round, num_clients_per_round))
                pairwise_metrics = []
                for i in range(len(cos_sim2)):
                    for j in range(i):
                        pairwise_metrics.append(cos_sim2[i][j])

                mean_pairwise_metrics = np.mean(pairwise_metrics)
                mean_pairwise_metrics_per_round.append(mean_pairwise_metrics)

            ax.plot(range(1, len(mean_pairwise_metrics_per_round)+1), mean_pairwise_metrics_per_round, color=color, label=strategy_name)

        ax.set_xlabel("Round", fontsize=14)
        ax.set_ylabel("Cosine similarity", fontsize=14)
        ax.legend(title='concentration', loc='upper right')
        ax.set_ylim(0, 1)
        ax.grid()
        if k == 0:
            ax.set_title("Random activations", fontsize=16)
            plt.savefig(experiments_dir / 'random_activations.png', dpi=300)
        else:
            ax.set_title("Test set activations", fontsize=16)
            plt.savefig(experiments_dir / 'test_activations.png', dpi=300)

def compare_clients_vs_global(strat, global_params, device, num_rounds=30, show_plot=False, axes=None):
    res_client, res_global_prev, res_global_next = [], [], []
    for round in range(num_rounds):
        client_models = strat.clients_models[round+1]
        _, global_model_prev = global_params[round]
        _, global_model_next = global_params[round+1]
        round_acts_client = []

        global_net_prev = set_model_parameters(network_generator_cnn(), global_model_prev)
        global_net_next = set_model_parameters(network_generator_cnn(), global_model_next)
        act_global_prev = get_activations_from_centralized_test(global_net_prev, device, centralized_mapping)
        act_global_next = get_activations_from_centralized_test(global_net_next, device, centralized_mapping)

        res_global_prev.append(act_global_prev)
        res_global_next.append(act_global_next)
        for _, client_params in client_models:
            client_net = set_model_parameters(network_generator_cnn(), client_params)
            act_client = get_activations_from_centralized_test(client_net, device, centralized_mapping)
            round_acts_client.append(act_client)  # softmaxed activations

        res_client.append(round_acts_client)

    cos_sim_prev, cos_sim_next, kld_prev, kld_next = [], [], [], []
    for round in range(num_rounds):
        client_round = res_client[round]
        global_prev_round = res_global_prev[round]
        global_next_round = res_global_next[round]

        cos_sim_prev_round = [cosine_sim(x, global_prev_round) for x in client_round]
        kld_prev_round = [entropy(x, global_prev_round) for x in client_round]
        cos_sim_next_round = [cosine_sim(x, global_next_round) for x in client_round]
        kld_next_round = [entropy(x, global_next_round) for x in client_round]

        cos_sim_prev.append(cos_sim_prev_round)
        cos_sim_next.append(cos_sim_next_round)
        kld_prev.append(kld_prev_round)
        kld_next.append(kld_next_round)

    if show_plot:
        x = np.arange(1, num_rounds+1)
        width = 0.35

        # Average Cosine Similarity
        axes[0].bar(x - width/2, [np.mean(round_cos_sim) for round_cos_sim in cos_sim_prev], width, label='previous')
        axes[0].bar(x + width/2, [np.mean(round_cos_sim) for round_cos_sim in cos_sim_next], width, label='next')
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Cosine Similarity')
        axes[0].set_ylim(0, 1)
        axes[0].legend(title='global model', loc='upper right')

        # Average KLD
        axes[1].bar(x - width/2, [np.mean(round_kld) for round_kld in kld_prev], width, label='previous')
        axes[1].bar(x + width/2, [np.mean(round_kld) for round_kld in kld_next], width, label='next')
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('KLD')

    return cos_sim_prev, cos_sim_next, kld_prev, kld_next

def clients_vs_global_plot(strategies, params, num_rounds=30):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    alphas = [r'$\alpha = 0.1$', r'$\alpha = 1.0$', r'$\alpha = 100.0$']
    for i, (alpha, strategy, param) in enumerate(zip(alphas, strategies, params)):
        cos_sim_prev, cos_sim_next, kld_prev, kld_next = compare_clients_vs_global(strategy, param, DEVICE, num_rounds=num_rounds, show_plot=True, axes=axes[i])

        axes[i, 0].set_title(alpha, fontsize=14, x=1.0, y=1.05, horizontalalignment='center')
        axes[i, 0].title.set_position((1.0, 1.05))
        axes[i, 1].set_ylim(0, max(max(kld_prev[0]), max(kld_next[0])) * 1.1)

        for ax in axes[i]:
            ax.grid(True, linestyle='--', alpha=0.7)

        if i == 2:
            for ax in axes[i]:
                ax.set_xlabel('Round', fontsize=12)
        else:
            for ax in axes[i]:
                ax.set_xlabel('')

    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    axes[0, 0].set_ylabel('Cosine Similarity', fontsize=12)
    axes[1, 0].set_ylabel('Cosine Similarity', fontsize=12)
    axes[2, 0].set_ylabel('Cosine Similarity', fontsize=12)
    axes[0, 1].set_ylabel('KLD', fontsize=12)
    axes[1, 1].set_ylabel('KLD', fontsize=12)
    axes[2, 1].set_ylabel('KLD', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(experiments_dir / 'clients_vs_global_prev_next.png', dpi=300)


def confusion_matrix_per_round(params):
    params01, params1, params100 = params
    for concentration in [0.1, 1, 100]:
        for round, clients in enumerate(clients_per_round[:10]):
                fig, axs = plt.subplots(1, 2, figsize=(20, 6))
                lda_partition = lda_partitions[str(concentration)]
                lda_flower_client_generator = get_flower_client_generator_cifar(model_generator=network_generator_cnn,
                    data_dir=data_dir,
                    partition_dir=lda_partition,
                )

                client_labels = {x: (lda_flower_client_generator(x)._load_dataset("train").data)['label'].values for x in clients}
                labels = sorted(set(label for labels in client_labels.values() for label in labels))

                matrix = np.zeros((len(clients), len(labels)))
                for i, client in enumerate(clients):
                    for label in client_labels[client]:
                        j = labels.index(label)
                        matrix[i, j] += 1

                sns.heatmap(matrix, annot=True, cmap="YlGnBu", fmt=".2f", xticklabels=labels, yticklabels=clients, ax=axs[0])
                axs[0].set_title(f'Client Distribution || $\\alpha = {concentration}$ || Round = {round+1}')
                axs[0].set_xlabel('Labels')
                axs[0].set_ylabel('Clients')

                if concentration == 0.1:
                    _, t_params01 = params01[round+1]
                    conf = compute_class_confusion(t_params01, network_generator_cnn, DEVICE, 'test')
                elif concentration == 1:
                    _, t_params1 = params1[round+1]
                    conf = compute_class_confusion(t_params1, network_generator_cnn, DEVICE, 'test')
                else:
                    _, t_params100 = params100[round+1]
                    conf = compute_class_confusion(t_params100, network_generator_cnn, DEVICE, 'test')

                sns.heatmap(conf, cmap='Blues', annot=True, fmt='d', ax=axs[1])
                axs[1].set_title(f'Confusion Matrix || $\\alpha = {concentration}$ || Round = {round+1}')
                axs[1].set_xlabel('Predicted')
                axs[1].set_ylabel('True')

                plt.tight_layout()
                plt.show()

def local_epochs_ablation(num_iterations=5, epoch_l=[1,2,5,10]):
    epoch_ablation = []
    for i in range(num_iterations):
        for num_epochs in epoch_l:
            for conc in [0.1, 1, 100]:
                print(f"NUM EPOCHS: {num_epochs} || CONC: {conc}")
                train_config['epochs'] = num_epochs
                def _on_fit_config_fn(server_round: int) -> dict[str, Scalar]:
                    return train_config | {"server_round": server_round}

                lda_partition = lda_partitions[str(conc)]
                lda_flower_client_generator = get_flower_client_generator_cifar(
                    model_generator=network_generator_cnn,
                    data_dir=data_dir,
                    partition_dir=lda_partition,
                )
                initial_parameters = ndarrays_to_parameters(get_model_parameters(pretrained_net))
                strategy = WrappedFedMed(
                    fraction_fit=fraction_fit,
                    fraction_evaluate=fraction_evaluate,
                    min_fit_clients=num_clients_per_round,
                    min_evaluate_clients=0,
                    min_available_clients=num_clients_per_round,
                    on_fit_config_fn=_on_fit_config_fn,
                    on_evaluate_config_fn=None,
                    evaluate_fn=federated_evaluation_function,
                    initial_parameters=initial_parameters,
                    accept_failures=False,
                    fit_metrics_aggregation_fn=aggregate_weighted_average,
                    evaluate_metrics_aggregation_fn=aggregate_weighted_average,
                )
                strategy.concentration = conc
                params, hist = start_seeded_simulation(
                    client_fn=lambda cid: lda_flower_client_generator(cid).to_client(),
                    num_clients=num_total_clients,
                    config=ServerConfig(num_rounds=1),
                    strategy=strategy,
                    name='ablation',
                    iteration=i
                )
                epoch_ablation.append({'iter': i, 'num_epochs': num_epochs, 'concentration': conc, 'params': params, 'hist': hist})
    torch.save(epoch_ablation, experiments_dir / 'epoch_ablation.pt')

def epoch_ablation_plot(epoch_ablation):
    # Group the data by concentration
    grouped_data = {}
    for item in epoch_ablation:
        concentration = item['concentration']
        iter = item['iter']
        if concentration not in grouped_data:
            grouped_data[concentration] = {'num_epochs': [], 'accuracy': [], 'sem': []}
        grouped_data[concentration]['num_epochs'].append(item['num_epochs'])
        grouped_data[concentration]['accuracy'].append(np.round(item['hist'].metrics_centralized['accuracy'][-1][-1], 2))

    line_styles = ['-', '--', '-.', ':']
    for i, concentration in enumerate(grouped_data):
        unique_epochs = sorted(list(set(grouped_data[concentration]['num_epochs'])))
        averaged_accuracy = []
        standard_errors = []
        for epoch in unique_epochs:
            epoch_accuracies = [acc for num_epoch, acc in zip(grouped_data[concentration]['num_epochs'], grouped_data[concentration]['accuracy']) if num_epoch == epoch]
            averaged_accuracy.append(np.mean(epoch_accuracies))
            standard_errors.append(np.std(epoch_accuracies) / np.sqrt(len(epoch_accuracies)))
        grouped_data[concentration]['num_epochs'] = [0] + unique_epochs
        grouped_data[concentration]['accuracy'] = [0.58] + averaged_accuracy
        grouped_data[concentration]['sem'] = [0] + standard_errors
        grouped_data[concentration]['line_style'] = line_styles[i % len(line_styles)]

    fig, ax = plt.subplots(figsize=(8, 6))
    for concentration, values in grouped_data.items():
        ax.errorbar(values['num_epochs'], values['accuracy'], yerr=values['sem'], label=f'$\\alpha$ = {concentration}', capsize=3, marker='o', linestyle=values['line_style'], markersize=4)

    # Add labels, title, and gridlines
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Local Epochs for Different LDA Concentrations')
    ax.legend()
    ax.grid(True)

    plt.savefig(experiments_dir / 'accuracy_vs_epochs.png', dpi=300)
    torch.save(grouped_data, experiments_dir / 'accuracy_vs_epochs.pt')

# Adjusting number of clients per round
def num_clients_ablation(clients_l=[5, 10, 25, 50, 100], num_iterations=3):
    epoch_ablation = []
    for i in range(num_iterations):
        for num_clients in clients_l:
            for conc in [0.1, 1, 100]:
                print(f"NUM CLIENTS: {num_clients} || CONC: {conc}")

                lda_partition = lda_partitions[str(conc)]
                lda_flower_client_generator = get_flower_client_generator_cifar(model_generator=network_generator_cnn,
                    data_dir=data_dir,
                    partition_dir=lda_partition,
                )

                fraction_fit: float = float(num_clients) / num_total_clients
                initial_parameters = ndarrays_to_parameters(get_model_parameters(pretrained_net))
                strategy = WrappedFedMed(
                    fraction_fit=fraction_fit,
                    fraction_evaluate=fraction_evaluate,
                    min_fit_clients=num_clients,
                    min_evaluate_clients=num_evaluate_clients,
                    min_available_clients=max(num_clients, num_evaluate_clients),
                    on_fit_config_fn=_on_fit_config_fn,
                    on_evaluate_config_fn=None,
                    evaluate_fn=federated_evaluation_function,
                    initial_parameters=initial_parameters,
                    accept_failures=False,
                    fit_metrics_aggregation_fn=aggregate_weighted_average,
                    evaluate_metrics_aggregation_fn=aggregate_weighted_average,
                )

                strategy.concentration = conc

                params, hist = start_seeded_simulation(
                    client_fn=lambda cid: lda_flower_client_generator(cid).to_client(),
                    num_clients=num_total_clients,
                    config=ServerConfig(num_rounds=1),
                    strategy=strategy,
                    name='ablation',
                    iteration=i
                )

                epoch_ablation.append({'iter': i, 'num_clients': num_clients, 'concentration': conc, 'params': params, 'hist': hist})
    torch.save(epoch_ablation, experiments_dir / 'num_clients_ablation.pt')


def num_clients_plot(epoch_ablation):
    grouped_data = {}
    for item in epoch_ablation:
        concentration = item['concentration']
        iter = item['iter']
        if concentration not in grouped_data:
            grouped_data[concentration] = {'num_clients': [], 'accuracy': [], 'sem': []}
        grouped_data[concentration]['num_clients'].append(item['num_clients'])
        grouped_data[concentration]['accuracy'].append(np.round(item['hist'].metrics_centralized['accuracy'][-1][-1], 2))

    line_styles = ['-', '--', '-.', ':']
    for i, concentration in enumerate(grouped_data):
        unique_epochs = sorted(list(set(grouped_data[concentration]['num_clients'])))
        averaged_accuracy = []
        standard_errors = []
        for epoch in unique_epochs:
            epoch_accuracies = [acc for num_epoch, acc in zip(grouped_data[concentration]['num_clients'], grouped_data[concentration]['accuracy']) if num_epoch == epoch]
            averaged_accuracy.append(np.mean(epoch_accuracies))
            standard_errors.append(np.std(epoch_accuracies) / np.sqrt(len(epoch_accuracies)))
        grouped_data[concentration]['num_clients'] = [0] + unique_epochs
        grouped_data[concentration]['accuracy'] = [0.58] + averaged_accuracy
        grouped_data[concentration]['sem'] = [0] + standard_errors
        grouped_data[concentration]['line_style'] = line_styles[i % len(line_styles)]

    fig, ax = plt.subplots(figsize=(8, 6))
    for concentration, values in grouped_data.items():
        ax.errorbar(values['num_clients'], values['accuracy'], yerr=values['sem'], label=f'$\\alpha$ = {concentration}', capsize=3, marker='o', linestyle=values['line_style'], markersize=4)

    ax.set_xlabel('Clients')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Clients Per Round for Different LDA Concentrations')
    ax.legend()
    ax.grid(True)

    plt.savefig(experiments_dir/ 'accuracy_vs_clients.png', dpi=300)
    torch.save(grouped_data, experiments_dir / 'accuracy_vs_clients.pt')


def majority_class_plot(params, num_rounds=30):
    params01, params1, params100 = params
    for i, concentration in enumerate([0.1, 1, 100]):
        fig, axs = plt.subplots(figsize=(8, 6))
        num_preds = []
        for round in range(1,1+num_rounds):
            clients = clients_per_round[round-1]
            lda_partition = lda_partitions[str(concentration)]
            lda_flower_client_generator = get_flower_client_generator_cifar(model_generator=network_generator_cnn,
                data_dir=data_dir,
                partition_dir=lda_partition,
            )

            client_labels = {x: (lda_flower_client_generator(x)._load_dataset("train").data)['label'].values for x in clients}
            labels = range(10)

            matrix = np.zeros((len(clients), len(labels)))
            for j, client in enumerate(clients):
                for label in client_labels[client]:
                    k = labels.index(label)
                    matrix[j, k] += 1

            if concentration == 0.1:
                _, t_params01 = params01[round]
                conf = compute_class_confusion(t_params01, network_generator_cnn, DEVICE, 'test')
            elif concentration == 1:
                _, t_params1 = params1[round]
                conf = compute_class_confusion(t_params1, network_generator_cnn, DEVICE, 'test')
            else:
                _, t_params100 = params100[round]
                conf = compute_class_confusion(t_params100, network_generator_cnn, DEVICE, 'test')

            pred_lab = torch.sum(conf, axis=0)
            client_labels = np.sum(matrix, axis=0)
            num_preds.append((pred_lab, client_labels))

        top_class = [np.argmax(y) for x, y in num_preds]
        pred = [x[y] for x,y in zip([x for x,y in num_preds], top_class)]
        client = [x[y] for x,y in zip([y for x,y in num_preds], top_class)]

        pred_lab_np = np.array(pred) / 1500
        client_labels_np = np.array(client) / 2500

        num_labels = len(pred_lab_np)
        x = np.arange(1, num_labels + 1)
        width = 0.35

        axs.bar(x - width/2, client_labels_np, width, label='Client Labels', hatch='///', edgecolor='black')
        axs.bar(x + width/2, pred_lab_np, width, label='Predicted Labels')
        axs.set_xticks(x)
        axs.set_xticklabels(x, rotation=45, ha='right')
        axs.set_ylabel('Proportion of Labels', fontsize=14)
        axs.set_title(f'$\\alpha = {concentration}$')
        axs.legend()
        axs.set_ylim(0, 1)
        axs.set_xlabel('Rounds', fontsize=14)
        plt.tight_layout()
        plt.savefig(experiments_dir / f'majority_class_{concentration}.png', dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose plots to generate')
    parser.add_argument('--generate_data', action='store_true', default=False,
                    help='Generate the data before plotting')
    parser.add_argument('--plot', type=str, required=True,
                        choices=['test_accuracy', 'clients_pairwise_similarity', 'clients_vs_global', 'local_epochs_ablation',
                                 'num_clients_ablation', 'majority_class'],
                        help='Select the plot to generate (default: all)')
    parser.add_argument('--num_rounds', type=int, default=10)
    parser.add_argument('--num_iterations', type=int, default=3)
    args = parser.parse_args()

    if args.plot in ['test_accuracy', 'clients_pairwise_similarity', 'clients_vs_global', 'majority_class']:
        if args.generate_data or not all([os.path.exists(weights_dir / f) for f in ['strat0.1_iters.pt', 'strat1_iters.pt', 'strat100_iters.pt']]):
            run_lda_experiments(num_rounds=args.num_rounds, num_clients_per_round=args.num_clients_per_round, num_iterations=args.num_iterations, num_evaluate_clients=0)

        res01, res1, res100 = torch.load(weights_dir / 'strat0.1_iters.pt'), torch.load(weights_dir / 'strat1_iters.pt'), torch.load(weights_dir / 'strat100_iters.pt')
        clients01, clients1, clients100 = res01[0]['clients_models'], res1[0]['clients_models'], res100[0]['clients_models'] # clients models (first iteration)
        params01, params1, params100 = res01[0]['params'], res1[0]['params'], res100[0]['params'] # global params (first iteration)

        strategies = [StratWrapper(0.1, clients01), StratWrapper(1, clients1), StratWrapper(100, clients100)]
        params = [params01, params1, params100]

        if args.plot == 'test_accuracy':
            plot_test_accuracy()
        elif args.plot == 'clients_pairwise_similarity':
            clients_models_activation_similarity(strategies, num_rounds=args.num_rounds)
        elif args.plot == 'clients_vs_global':
            clients_vs_global_plot(strategies, params, num_rounds=args.num_rounds)
        elif args.plot == 'majority_class':
            majority_class_plot(params, num_rounds=args.num_rounds)

    elif args.plot == 'local_epochs_ablation':
        if args.generate_data or not os.path.exists(experiments_dir / 'epoch_ablation.pt'):
             local_epochs_ablation(num_iterations=args.num_iterations)
        epoch_ablation = torch.load(experiments_dir / 'epoch_ablation.pt')
        epoch_ablation_plot(epoch_ablation)

    elif args.plot == 'num_clients_ablation':
        if args.generate_data or not os.path.exists(experiments_dir / 'num_clients_ablation.pt'):
             num_clients_ablation(num_iterations=args.num_iterations)
        client_ablation = torch.load(experiments_dir / 'num_clients_ablation.pt')
        num_clients_plot(client_ablation)
    
    else:
        raise Exception(f'Plot type {args.plot} not supported')
