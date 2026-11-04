import argparse
import io
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data_utils import read_manifest


DEFAULT_ENCODER = 'speechbrain/spkrec-xvect-voxceleb'


def build_parser():
    parser = argparse.ArgumentParser(
        description='Fit or evaluate the paper Mahalanobis one-class baseline.'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    fit_parser = subparsers.add_parser('fit', help='fit bonafide Gaussian statistics')
    fit_parser.add_argument('--train-list', required=True)
    fit_parser.add_argument('--output-stats', required=True)
    fit_parser.add_argument('--audio-root', default=None)
    add_encoder_arguments(fit_parser)

    eval_parser = subparsers.add_parser('evaluate', help='score a binary evaluation list')
    eval_parser.add_argument('--eval-list', required=True)
    eval_parser.add_argument('--stats', required=True)
    eval_parser.add_argument('--scores', required=True)
    eval_parser.add_argument('--summary', required=True)
    eval_parser.add_argument('--audio-root', default=None)
    add_encoder_arguments(eval_parser)
    return parser


def add_encoder_arguments(parser):
    parser.add_argument('--encoder-source', default=DEFAULT_ENCODER)
    parser.add_argument('--encoder-cache', default='checkpoints/spkrec-xvect-voxceleb')
    parser.add_argument('--device', default=None, help='device override, e.g. cuda:0')


def load_encoder(args, device):
    from speechbrain.inference.speaker import EncoderClassifier

    return EncoderClassifier.from_hparams(
        source=args.encoder_source,
        savedir=args.encoder_cache,
        run_opts={'device': device},
    )


def extract_embedding(encoder, audio_path, device):
    import torchaudio

    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        raise ValueError(f'{audio_path}: expected 16 kHz audio, found {sample_rate} Hz')
    with torch.no_grad():
        embedding = encoder.encode_batch(waveform.to(device))
    return embedding.squeeze(0).squeeze(0).cpu().numpy()


def fit(args):
    from sklearn.covariance import LedoitWolf

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    entries = read_manifest(args.train_list, args.audio_root)
    non_bonafide = [label for _, label in entries if label != 'bonafide']
    if non_bonafide:
        raise ValueError('the Mahalanobis fitting manifest must contain only bonafide audio')

    encoder = load_encoder(args, device)
    embeddings = [
        extract_embedding(encoder, path, device)
        for path, _ in tqdm(entries, desc='Extracting bonafide x-vectors')
    ]
    embeddings = np.stack(embeddings)
    stats = {
        'all': {
            'mean': torch.from_numpy(embeddings.mean(axis=0, keepdims=True)),
            'cov': LedoitWolf().fit(embeddings).covariance_,
        }
    }

    output_path = Path(args.output_stats)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('wb') as handle:
        pickle.dump(stats, handle)
    print(f'wrote statistics for {len(embeddings)} bonafide files to {output_path}')


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_gaussian(path):
    with Path(path).open('rb') as handle:
        models = CPUUnpickler(handle).load()
    if len(models) != 1:
        raise ValueError('expected one global bonafide Gaussian in the statistics file')
    stats = next(iter(models.values()))
    mean = as_numpy(stats['mean']).squeeze()
    covariance = as_numpy(stats['cov'])
    try:
        precision = np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(covariance)
    return mean, precision


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda data: torch.load(
                io.BytesIO(data), map_location='cpu'
            )
        return super().find_class(module, name)


def compute_eer(labels, scores):
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    index = np.argmin(np.abs(fpr - (1 - tpr)))
    return float(fpr[index]), float(thresholds[index])


def evaluate(args):
    from scipy.spatial.distance import mahalanobis

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    entries = read_manifest(args.eval_list, args.audio_root)
    mean, precision = load_gaussian(args.stats)
    encoder = load_encoder(args, device)

    labels = []
    score_rows = []
    for audio_path, label in tqdm(entries, desc='Scoring Mahalanobis distance'):
        if label not in {'bonafide', 'spoof'}:
            raise ValueError(f'unsupported evaluation label: {label}')
        numeric_label = 0 if label == 'bonafide' else 1
        embedding = extract_embedding(encoder, audio_path, device)
        score = float(mahalanobis(embedding, mean, precision))
        labels.append(numeric_label)
        score_rows.append((audio_path, numeric_label, score))

    scores = np.asarray([row[2] for row in score_rows])
    eer, threshold = compute_eer(np.asarray(labels), scores)

    scores_path = Path(args.scores)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open('w') as handle:
        for audio_path, label, score in score_rows:
            handle.write(f'{audio_path} {label} {score:.6f}\n')

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = f'EER: {eer:.4f} at threshold {threshold:.6f}'
    summary_path.write_text(summary + '\n')
    print(summary)


def main():
    args = build_parser().parse_args()
    if args.command == 'fit':
        fit(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
