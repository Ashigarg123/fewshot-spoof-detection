import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from kshot_dataset_ssl import Kshot_Dataset


def build_eval_parser(description, default_support_sizes):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--eval-list', required=True, help='evaluation audio/label list')
    parser.add_argument('--checkpoint', required=True, help='model checkpoint')
    parser.add_argument('--ssl-checkpoint', required=True, help='XLS-R checkpoint')
    parser.add_argument('--audio-root', default=None,
                        help='root prepended to relative manifest paths')
    parser.add_argument('--output-dir', required=True, help='EER output directory')
    parser.add_argument('--tag', default=None, help='output filename tag')
    parser.add_argument('--support-sizes', type=int, nargs='+', default=default_support_sizes)
    parser.add_argument('--num-iters', type=int, default=100)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', default=None, help='device override, e.g. cuda:0')
    return parser


def load_model(model_class, args, device, model_kwargs=None):
    model = model_class(device, args, **(model_kwargs or {})).to(device)
    state = torch.load(args.checkpoint, map_location=torch.device(device))
    model.load_state_dict(state)
    model.eval()
    return model


def extract_embeddings(list_path, model, device, audio_root=None):
    dataset = Kshot_Dataset(list_path, audio_root)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    real_embeddings = []
    spoof_embeddings = []

    with torch.no_grad():
        for features, labels in tqdm(dataloader):
            embeddings = model(features.to(device)).squeeze(1).cpu()
            for index, label in enumerate(labels.cpu()):
                if label.item() == 0:
                    real_embeddings.append(embeddings[index])
                elif label.item() == 1:
                    spoof_embeddings.append(embeddings[index])

    if not real_embeddings or not spoof_embeddings:
        raise ValueError('evaluation list must contain bonafide and spoof examples')
    return torch.stack(real_embeddings).numpy(), torch.stack(spoof_embeddings).numpy()


def euclidean_dist(queries, prototypes):
    queries = torch.as_tensor(queries, dtype=torch.float32)
    prototypes = torch.as_tensor(prototypes, dtype=torch.float32)
    if queries.size(1) != prototypes.size(1):
        raise ValueError('query and prototype dimensions do not match')
    return torch.pow(queries[:, None, :] - prototypes[None, :, :], 2).sum(2)


def compute_eer(target_scores, nontarget_scores):
    count = target_scores.size + nontarget_scores.size
    scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))
    indices = np.argsort(scores, kind='mergesort')
    labels = labels[indices]

    target_sums = np.cumsum(labels)
    nontarget_sums = nontarget_scores.size - (np.arange(1, count + 1) - target_sums)
    frr = np.concatenate((np.atleast_1d(0), target_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_sums / nontarget_scores.size))
    thresholds = np.concatenate((np.atleast_1d(scores[indices[0]] - 0.001), scores[indices]))
    index = np.argmin(np.abs(frr - far))
    return np.mean((frr[index], far[index])), thresholds[index]


def evaluate_support_size(real_embeddings, spoof_embeddings, support_size,
                          num_iters, choose_indices, make_prototype):
    results = []
    for _ in range(num_iters):
        real_indices = choose_indices(len(real_embeddings), support_size)
        spoof_indices = choose_indices(len(spoof_embeddings), support_size)
        prototypes = np.stack((
            make_prototype(real_embeddings[real_indices]),
            make_prototype(spoof_embeddings[spoof_indices]),
        ))

        real_query = np.delete(real_embeddings, real_indices, axis=0)
        spoof_query = np.delete(spoof_embeddings, spoof_indices, axis=0)
        queries = np.concatenate((real_query, spoof_query))
        labels = np.array([0] * len(real_query) + [1] * len(spoof_query))
        scores = F.softmax(-euclidean_dist(queries, prototypes), dim=1)[:, 0]
        eer, threshold = compute_eer(
            scores[labels == 0].numpy(), scores[labels == 1].numpy()
        )
        results.append((eer, float(threshold)))
    return results


def run_evaluation(args, model, choose_indices, make_prototype):
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    real_embeddings, spoof_embeddings = extract_embeddings(
        args.eval_list, model, device, args.audio_root
    )
    print(f'embeddings ready: bona={len(real_embeddings)} spoof={len(spoof_embeddings)}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or Path(args.eval_list).stem

    for support_size in args.support_sizes:
        if support_size >= len(real_embeddings) or support_size >= len(spoof_embeddings):
            raise ValueError(f'support size {support_size} exceeds available examples')
        results = evaluate_support_size(
            real_embeddings,
            spoof_embeddings,
            support_size,
            args.num_iters,
            choose_indices,
            make_prototype,
        )
        mean_eer = np.mean([eer for eer, _ in results])
        output_path = output_dir / f'{support_size}_support_{tag}-eer.txt'
        with output_path.open('w') as handle:
            for index, (eer, threshold) in enumerate(results):
                handle.write(f'Iter {index}: EER {eer:.4f} @ thresh {threshold:.4f}\n')
            handle.write(f'Mean EER {mean_eer:.4f}\n')
        print(f'support={support_size} mean_eer={mean_eer:.4f} wrote={output_path}')
