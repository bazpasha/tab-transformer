import argparse
import itertools
import time
import shutil
import random
import numpy as np

import torch
import torch.nn.functional as F

from qhoptim.pyt import QHAdam

from trainer import Trainer, train, get_default_optimizer_params, train_tab_transformer
from data import get_dataset
from model import TabTransformer, get_random_masks, get_tree_masks, get_window_masks
from utils import get_attention_function


def parse_attention_functions(attention_function):
    if ":" in attention_function:
        return attention_function.split(":", 1)
    return attention_function, attention_function


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--dataset-seed", type=int, default=0, help="Dataset split and transform seed")
    parser.add_argument("--n-tokens", type=int, nargs="+", required=True, help="Number of tokens")
    parser.add_argument("--n-transformers", type=int, nargs="+", required=True, help="Number of transformers")
    parser.add_argument("--d-model", type=int, nargs="+", required=True, help="Model dimensionality")
    parser.add_argument("--attention-function", type=str, nargs="+", required=True, help="Attention function")
    parser.add_argument("--model-seed", type=int, default=0, help="Model training seed")
    parser.add_argument("--batch-size", type=int, default=1024, help="Train batch size")
    parser.add_argument("--report-frequency", type=int, default=100, help="Report and averaging frequency")
    parser.add_argument("--n-heads", type=int, default=1, nargs="+", help="Number of heads in attention")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--cuda", action="store_true", help="Use GPU if available")
    parser.add_argument("--standardize", action="store_true", help="Standardize target for regression")
    parser.add_argument("--epochs", type=int, help="Fixed number of epochs for debug")
    parser.add_argument("--output-dir", type=str, help="Output directory with all the experiments")
    parser.add_argument("--mask", type=str, nargs="+", help="Mask names")
    parser.add_argument("--pretrain", action="store_true", help="Pretrain model")
    parser.add_argument("--mask-fraction", type=float, default=0.1, help="Mask fraction for pretrain")
    return parser.parse_args()


def main():
    args = get_args()

    if args.cuda and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    dataset = get_dataset(
        dataset=args.dataset,
        standardize=args.standardize,
        seed=args.dataset_seed,
    )

    time_suffix = '{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}'.format(*time.gmtime()[:5])
    search_space = itertools.product(
        args.n_tokens,
        args.n_transformers,
        args.d_model,
        args.attention_function,
        args.n_heads,
        args.mask or [None]
    )

    for n_tokens, n_transformers, d_model, attention_function, n_heads, mask in search_space:
        experiment_name = "{}.{}.{}.{}.{}.{}.{}_{}".format(
            args.dataset,
            n_tokens,
            n_transformers,
            d_model,
            attention_function,
            n_heads,
            mask or "full",
            time_suffix
        )

        train_tab_transformer(
            n_tokens=n_tokens,
            n_transformers=n_transformers,
            d_model=d_model,
            attention_function=attention_function,
            n_heads=n_heads,
            experiment_name=experiment_name,
            dataset=dataset,
            batch_size=args.batch_size,
            device=device,
            report_frequency=args.report_frequency,
            mask=mask,
            epochs=args.epochs or float("inf"),
            output_dir=args.output_dir,
            model_seed=args.model_seed,
            verbose=args.verbose,
            pretrain=args.pretrain,
            mask_fraction=args.mask_fraction
        )


if __name__ == "__main__":
    main()