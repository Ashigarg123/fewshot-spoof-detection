import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fewshot_eval import compute_eer
from kshot_dataset_ssl import Kshot_Dataset as BinaryDataset
from kshot_multi_aug import Kshot_Dataset as MultiClassDataset
from ssl_aasist import Model
from training_args import add_rawboost_arguments
from training_utils import init_seed


MULTICLASS_LABELS = {
    'A01': 0,
    'bonafide': 1,
    'A02': 2,
    'A03': 3,
    'A04': 4,
    'A05': 5,
    'A06': 6,
}


def build_parser():
    parser = argparse.ArgumentParser(
        description='Train or evaluate zero-shot SSL-AASIST classifiers.'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_parser = subparsers.add_parser(
        'train-multiclass',
        help='train the paper seven-class ASVspoof 2019 classifier',
    )
    add_common_arguments(train_parser)
    train_parser.add_argument('--train-list', required=True)
    train_parser.add_argument('--valid-list', required=True)
    train_parser.add_argument('--output-dir', required=True)
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--batch-size', type=int, default=64)
    train_parser.add_argument('--learning-rate', type=float, default=1e-5)
    train_parser.add_argument('--weight-decay', type=float, default=1e-4)
    train_parser.add_argument('--num-workers', type=int, default=8)
    train_parser.add_argument('--manual-seed', type=int, default=1234)
    add_rawboost_arguments(train_parser)
    train_parser.set_defaults(algo=5)

    eval_parser = subparsers.add_parser('evaluate', help='score a classifier checkpoint')
    add_common_arguments(eval_parser)
    eval_parser.add_argument('--model', choices=('binary', 'multiclass'), required=True)
    eval_parser.add_argument('--checkpoint', required=True)
    eval_parser.add_argument('--eval-list', required=True)
    eval_parser.add_argument('--scores', required=True)
    eval_parser.add_argument('--summary', required=True)
    eval_parser.add_argument('--batch-size', type=int, default=10)
    return parser


def add_common_arguments(parser):
    parser.add_argument('--ssl-checkpoint', required=True)
    parser.add_argument('--audio-root', default=None)
    parser.add_argument('--device', default=None)


def build_model(args, device, num_classes, freeze_ssl):
    return Model(
        device,
        args,
        use_attention=False,
        num_classes=num_classes,
        freeze_ssl=freeze_ssl,
    ).to(device)


def run_epoch(model, dataloader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0

    for features, labels in tqdm(dataloader, leave=False):
        features = features.to(device)
        labels = labels.long().to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits, _ = model(features)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(labels)
        total_examples += len(labels)
    return total_loss / total_examples


def train_multiclass(args):
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    init_seed(args)
    train_set = MultiClassDataset(
        args.train_list,
        args,
        args.algo,
        label_map=MULTICLASS_LABELS,
        audio_backend='torchaudio',
    )
    valid_set = MultiClassDataset(
        args.valid_list,
        args,
        0,
        is_train=False,
        label_map=MULTICLASS_LABELS,
        audio_backend='torchaudio',
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = build_model(args, device, 7, freeze_ssl=False)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'training.tsv').open('w') as log:
        log.write('epoch\ttrain_loss\tvalid_loss\n')
        for epoch in range(args.epochs):
            train_loss = run_epoch(model, train_loader, device, optimizer)
            valid_loss = run_epoch(model, valid_loader, device)
            print(
                f'epoch={epoch} train_loss={train_loss:.6f} '
                f'valid_loss={valid_loss:.6f}'
            )
            log.write(f'{epoch}\t{train_loss}\t{valid_loss}\n')
            log.flush()
            torch.save(model.state_dict(), output_dir / f'epoch_{epoch}.pth')


def evaluate(args):
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_classes = 2 if args.model == 'binary' else 7
    model = build_model(args, device, num_classes, freeze_ssl=True)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dataset = BinaryDataset(args.eval_list, args.audio_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    rows = []
    with torch.no_grad():
        for features, labels in tqdm(dataloader):
            logits, _ = model(features.to(device))
            if args.model == 'binary':
                # Upstream SSL-AASIST assigns class 1 to bonafide audio.
                scores = logits[:, 1]
            else:
                spoof_classes = [0, 2, 3, 4, 5, 6]
                scores = logits[:, spoof_classes].mean(dim=1)
            for label, score in zip(labels, scores.cpu()):
                rows.append((dataset.x[len(rows)], int(label), float(score)))

    labels = np.asarray([row[1] for row in rows])
    scores = np.asarray([row[2] for row in rows])
    if args.model == 'binary':
        eer, threshold = compute_eer(scores[labels == 0], scores[labels == 1])
    else:
        eer, threshold = compute_eer(scores[labels == 1], scores[labels == 0])

    scores_path = Path(args.scores)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open('w') as handle:
        for audio_path, label, score in rows:
            label_name = 'bonafide' if label == 0 else 'spoof'
            handle.write(f'{audio_path} {label_name} {score:.8f}\n')

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = f'EER: {eer:.6f} at threshold {threshold:.8f}'
    summary_path.write_text(summary + '\n')
    print(summary)


def main():
    args = build_parser().parse_args()
    if args.command == 'train-multiclass':
        train_multiclass(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
