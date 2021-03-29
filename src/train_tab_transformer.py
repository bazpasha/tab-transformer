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
from model import TabTransformer
from utils import get_attention_function


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
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--cuda", action="store_true", help="Use GPU if available")
    parser.add_argument("--standardize", action="store_true", help="Standardize target for regression")
    parser.add_argument("--epochs", type=int, help="Fixed number of epochs for debug")
    parser.add_argument("--output-dir", type=str, help="Output directory with all the experiments")
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

    for n_tokens, n_transformers, d_model, attention_function in \
            itertools.product(args.n_tokens, args.n_transformers, args.d_model, args.attention_function):

        torch.manual_seed(args.model_seed)
        random.seed(args.model_seed)
        np.random.seed(args.model_seed)

        experiment_name = "{}.{}.{}.{}.{}_{}".format(
            args.dataset, n_tokens, n_transformers, d_model, attention_function, time_suffix
        )

        attention_kwargs = {"attention_function": get_attention_function(attention_function)}

        model = TabTransformer(
            n_features=dataset.n_features,
            n_tokens=n_tokens,
            d_model=d_model,
            n_transformers=n_transformers,
            dim_feedforward=2 * d_model,
            dim_output=dataset.dim_output,
            attention_kwargs=attention_kwargs,
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