import argparse
import itertools
import time
import shutil
import random
import numpy as np

import torch
import torch.nn.functional as F

from qhoptim.pyt import QHAdam

from trainer import Trainer, train, get_default_optimizer_params
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
        torch.manual_seed(args.model_seed)
        random.seed(args.model_seed)
        np.random.seed(args.model_seed)

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

        agg_attention_function, attention_function = parse_attention_functions(attention_function)

        agg_attention_kwargs = {
            "attention_function": get_attention_function(agg_attention_function),
            "n_heads": n_heads,
        }

        attention_kwargs = {
            "attention_function": get_attention_function(attention_function),
            "n_heads": n_heads,
        }

        masks = None
        if mask is not None:
            if mask == "random":
                masks = get_random_masks(n_tokens, n_transformers, n_active=5)
            elif mask == "window":
                masks = get_window_masks(n_tokens, n_transformers, window_size=3)
            elif mask == "tree":
                masks = get_tree_masks(n_tokens, n_transformers)
            else:
                assert mask == "full"

        model = TabTransformer(
            n_features=dataset.n_features,
            n_tokens=n_tokens,
            d_model=d_model,
            n_transformers=n_transformers,
            dim_feedforward=2 * d_model,
            dim_output=dataset.dim_output,
            attention_kwargs=attention_kwargs,
            agg_attention_kwargs=agg_attention_kwargs,
            masks=masks,
        ).to(device)

        trainer = Trainer(
            model=model,
            loss_function=F.mse_loss if dataset.dataset_task == "regression" else F.cross_entropy,
            experiment_name=experiment_name,
            warm_start=False,
            Optimizer=QHAdam,
            optimizer_params=get_default_optimizer_params(),
            verbose=args.verbose,
            n_last_checkpoints=5
        )

        train(
            trainer=trainer,
            data=dataset.data,
            batch_size=args.batch_size,
            device=device,
            report_frequency=args.report_frequency,
            dataset_task=dataset.dataset_task,
            epochs=args.epochs or float("inf"),
            n_classes=dataset.n_classes,
            targets_std=dataset.target_std,
            verbose=args.verbose,
        )

        if args.output_dir is not None:
            shutil.move(trainer.experiment_path, args.output_dir)


if __name__ == "__main__":
    main()