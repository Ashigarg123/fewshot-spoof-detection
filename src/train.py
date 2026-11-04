import argparse
import os
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

from episodic import EpisodicBatchSampler, prototypical_loss
from kshot_dataset_ssl import Kshot_Dataset as BinaryDataset
from kshot_multi_aug import Kshot_Dataset as MultiClassDataset
from ssl_aasist import Model
from training_args import add_rawboost_arguments, build_training_parser
from training_utils import (
    build_episode_dataloader,
    init_lr_scheduler,
    init_optim,
    init_seed,
    save_list_to_file,
    save_training_plots,
)


PROFILES = {
    'binary_mean': {
        'output': 'results/binary_mean_5shot',
        'iterations': 30,
        'classes': 2,
        'set_batches': True,
        'attention': False,
        'freeze_ssl': False,
        'multiclass': False,
    },
    'binary_attention': {
        'output': 'results/attn_binary_5shot',
        'iterations': 100,
        'classes': 2,
        'set_batches': False,
        'attention': True,
        'freeze_ssl': True,
        'multiclass': False,
    },
    'multiclass_attention': {
        'output': 'results/multiclass_attention_5shot',
        'iterations': 100,
        'classes': 3,
        'set_batches': False,
        'attention': True,
        'freeze_ssl': False,
        'multiclass': True,
    },
}


def parse_args():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--model', choices=PROFILES, default='binary_attention')
    selected, _ = selector.parse_known_args()
    profile = PROFILES[selected.model]

    parser = build_training_parser(
        profile['output'],
        iterations=profile['iterations'],
        train_classes=profile['classes'],
        include_set_batches=profile['set_batches'],
    )
    parser.add_argument('--model', choices=PROFILES, default=selected.model)
    if profile['multiclass']:
        add_rawboost_arguments(parser)
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, options, prototype_aggregator):
    losses = []
    accuracies = []
    model.train()
    for features, targets in tqdm(dataloader):
        optimizer.zero_grad()
        embeddings = model(features.to(options.device))
        loss, accuracy, _, _ = prototypical_loss(
            embeddings,
            targets.to(options.device),
            options.num_support_tr,
            prototype_aggregator,
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accuracies.append(accuracy.item())
    return (
        np.mean(losses[-options.iterations:]),
        np.mean(accuracies[-options.iterations:]),
        losses,
        accuracies,
    )


def validate(model, dataloader, options, prototype_aggregator):
    losses = []
    accuracies = []
    model.eval()
    with torch.no_grad():
        for features, targets in dataloader:
            embeddings = model(features.to(options.device))
            loss, accuracy, _, _ = prototypical_loss(
                embeddings,
                targets.to(options.device),
                options.num_support_val,
                prototype_aggregator,
            )
            losses.append(loss.item())
            accuracies.append(accuracy.item())
    return (
        np.mean(losses[-options.iterations:]),
        np.mean(accuracies[-options.iterations:]),
        losses,
        accuracies,
    )


def build_components(options, profile):
    model = Model(options.device, options, use_attention=profile['attention'])
    if profile['freeze_ssl']:
        for parameter in model.ssl_model.model.parameters():
            parameter.requires_grad = False
    model = model.to(options.device)

    if profile['multiclass']:
        dataset_class = MultiClassDataset
        dataset_args = (options, options.algo)
        sampler_class = partial(EpisodicBatchSampler, always_include_class_zero=True)
    else:
        dataset_class = BinaryDataset
        dataset_args = (options.audio_root,)
        sampler_class = EpisodicBatchSampler

    train_loader = build_episode_dataloader(
        options.train_list, options, 'train', dataset_class, sampler_class, dataset_args
    )
    valid_loader = build_episode_dataloader(
        options.valid_list, options, 'val', dataset_class, sampler_class, dataset_args
    )
    aggregator = model.attention_aggregation if profile['attention'] else None
    return model, train_loader, valid_loader, aggregator


def main():
    options = parse_args()
    os.makedirs(options.experiment_root, exist_ok=True)
    init_seed(options)
    options.device = 'cuda:0' if torch.cuda.is_available() and options.cuda else 'cpu'
    profile = PROFILES[options.model]
    model, train_loader, valid_loader, aggregator = build_components(options, profile)
    optimizer = init_optim(options, model)
    scheduler = init_lr_scheduler(options, optimizer)

    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    epoch_train_losses = []
    epoch_train_accuracies = []
    epoch_valid_losses = []
    epoch_valid_accuracies = []
    best_accuracy = float('-inf')

    for epoch in range(options.epochs):
        train_loss, train_accuracy, batch_losses, batch_accuracies = train_epoch(
            model, train_loader, optimizer, options, aggregator
        )
        valid_loss, valid_accuracy, batch_valid_losses, batch_valid_accuracies = validate(
            model, valid_loader, options, aggregator
        )
        scheduler.step()
        train_losses.extend(batch_losses)
        train_accuracies.extend(batch_accuracies)
        valid_losses.extend(batch_valid_losses)
        valid_accuracies.extend(batch_valid_accuracies)
        epoch_train_losses.append(train_loss)
        epoch_train_accuracies.append(train_accuracy)
        epoch_valid_losses.append(valid_loss)
        epoch_valid_accuracies.append(valid_accuracy)
        print(
            f'epoch={epoch} train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} '
            f'val_loss={valid_loss:.4f} val_acc={valid_accuracy:.4f}'
        )

        state = model.state_dict()
        torch.save(state, os.path.join(options.experiment_root, f'epoch_{epoch}.pth'))
        if valid_accuracy >= best_accuracy:
            torch.save(state, os.path.join(options.experiment_root, 'best_model.pth'))
            best_accuracy = valid_accuracy

    torch.save(model.state_dict(), os.path.join(options.experiment_root, 'last_model.pth'))
    save_training_plots(
        options.experiment_root,
        epoch_train_losses,
        epoch_valid_losses,
        epoch_train_accuracies,
        epoch_valid_accuracies,
    )
    for name, values in (
        ('train_loss', train_losses),
        ('train_acc', train_accuracies),
        ('val_loss', valid_losses),
        ('val_acc', valid_accuracies),
    ):
        save_list_to_file(os.path.join(options.experiment_root, f'{name}.txt'), values)


if __name__ == '__main__':
    main()
