# Reproducing L361 plots/experiments

First, we must load the CIFAR-10 dataset and generate the federated LDA partitions (0.1, 1, 100) for 100 clients.
This creates the cifar directory, with data stored in ```cifar/client_data_mappings```. The experiment results are stored in ``cifar/experiments`` while pretrained weights are stored in ``cifar/weights``.

1.  Load CIFAR-10 dataset and create federated LDA partitions (0.1, 1, 100)
    - ```python cifar.py ```

2.  Pretrain a model on the centralised test set and save it down to cifar/weights
    - ```python centralised_pretraining.py ```

3.  Run experiments and generate plots - see plots.py for all options. We need to firstly generate the data for each experiment, then we can plot; subsequently we can choose to regenerate the data with the --generate flag.
    - ```python plots.py --plot test_accuracy```
    - ```python plots.py --plot local_epochs_ablation```
    - ```python plots.py --plot test_accuracy --generate_data --num_rounds 10```

