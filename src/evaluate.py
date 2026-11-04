import argparse

import numpy as np
import torch

from fewshot_eval import build_eval_parser, load_model, run_evaluation
from ssl_aasist import Model


PROFILES = {
    'binary_mean': {'support_sizes': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]},
    'binary_attention': {'support_sizes': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]},
    'multiclass_attention': {'support_sizes': [1, 5, 10]},
}


def attention_prototype(support, model, device):
    tensor = torch.as_tensor(support, dtype=torch.float32, device=device).unsqueeze(1)
    with torch.no_grad():
        return model.attention_aggregation(tensor).squeeze(0).cpu().numpy()


def parse_args():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--model', choices=PROFILES, default='binary_attention')
    selected, _ = selector.parse_known_args()
    parser = build_eval_parser(
        'Evaluate a few-shot SSL-AASIST checkpoint.',
        PROFILES[selected.model]['support_sizes'],
    )
    parser.add_argument('--model', choices=PROFILES, default=selected.model)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    uses_attention = args.model != 'binary_mean'
    model = load_model(
        Model,
        args,
        device,
        model_kwargs={'use_attention': uses_attention},
    )

    if args.model == 'binary_attention':
        rng = np.random.default_rng(args.seed)
        choose_indices = lambda count, size: rng.choice(count, size, replace=False)
        make_prototype = lambda support: attention_prototype(support, model, device)
    else:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        choose_indices = lambda count, size: np.random.choice(count, size, replace=False)
        make_prototype = lambda support: np.mean(support, axis=0)

    run_evaluation(args, model, choose_indices, make_prototype)


if __name__ == '__main__':
    main()
