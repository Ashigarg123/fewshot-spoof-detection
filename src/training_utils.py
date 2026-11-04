import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def init_seed(options):
    torch.cuda.cudnn_enabled = False
    random.seed(options.manual_seed)
    np.random.seed(options.manual_seed)
    torch.manual_seed(options.manual_seed)
    torch.cuda.manual_seed(options.manual_seed)


def init_optim(options, model):
    return torch.optim.Adam(model.parameters(), lr=options.learning_rate)


def init_lr_scheduler(options, optimizer):
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        gamma=options.lr_scheduler_gamma,
        step_size=options.lr_scheduler_step,
    )


def save_list_to_file(path, values):
    with open(path, 'w') as handle:
        for value in values:
            handle.write(f'{value}\n')


def build_episode_dataloader(list_path, options, mode, dataset_class,
                             sampler_class, dataset_args=()):
    dataset = dataset_class(list_path, *dataset_args)
    classes = options.classes_per_it_tr if mode == 'train' else options.classes_per_it_val
    support = options.num_support_tr if mode == 'train' else options.num_support_val
    query = options.num_query_tr if mode == 'train' else options.num_query_val
    sampler = sampler_class(
        labels=list(dataset.y),
        classes_per_it=classes,
        num_samples=support + query,
        iterations=options.iterations,
    )
    return torch.utils.data.DataLoader(dataset, batch_sampler=sampler)


def save_training_plots(output_dir, train_loss, valid_loss, train_acc, valid_acc):
    os.makedirs(output_dir, exist_ok=True)
    for train_values, valid_values, title, ylabel, filename in (
        (train_loss, valid_loss, 'Training and Validation Loss', 'Loss', 'train_val_loss.png'),
        (train_acc, valid_acc, 'Training and Validation Accuracy', 'Accuracy', 'train_val_acc.png'),
    ):
        epochs = range(1, len(train_values) + 1)
        figure = plt.figure(figsize=(12, 4))
        plt.plot(epochs, train_values, label=f'Training {ylabel.lower()}')
        plt.plot(epochs, valid_values, label=f'Validation {ylabel.lower()}')
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.legend()
        figure.savefig(os.path.join(output_dir, filename))
        plt.close(figure)
