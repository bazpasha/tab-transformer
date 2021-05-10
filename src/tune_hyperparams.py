import argparse
import time
import torch

from ax import ChoiceParameter, RangeParameter, ParameterType, SearchSpace, Models, OrderConstraint
from ax.modelbridge.factory import get_sobol

from trainer import train_catboost, train_fcn
from data import get_dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--model-name", type=str, help="Model name")
    parser.add_argument("--params-seed", type=int, default=0, help="Ax seed")
    parser.add_argument("--model-seed", type=int, default=0, help="Model seed")
    parser.add_argument("--dataset-seed", type=int, default=1337, help="Dataset split and transform seed")
    parser.add_argument("--cuda", action="store_true", help="Use GPU if available")
    parser.add_argument("--n-sweeps", type=int, default=30, help="Number of sweeps")
    parser.add_argument("--standardize", action="store_true", help="Standardize target for regression")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", type=str, help="Output directory with all the experiments")
    parser.add_argument("--skip-sweeps", type=int)

    return parser.parse_args()


def tune_catboost(
    n_sweeps,
    time_suffix,
    dataset_name,
    dataset,
    use_gpu,
    output_dir,
    model_seed,
    params_seed,
    verbose,
    skip_sweeps=None,
):
    search_space = SearchSpace(parameters=[
        RangeParameter(name="learning_rate", parameter_type=ParameterType.FLOAT, lower=1e-4, upper=1.0, log_scale=True),
        RangeParameter(name="l2_leaf_reg", parameter_type=ParameterType.FLOAT, lower=1, upper=10),
        RangeParameter(name="bagging_temperature", parameter_type=ParameterType.FLOAT, lower=0, upper=1),
        RangeParameter(name="leaf_estimation_iterations", parameter_type=ParameterType.INT, lower=1, upper=10),
        RangeParameter(name="random_strength", parameter_type=ParameterType.INT, lower=1, upper=20),
    ])

    sobol = get_sobol(search_space=search_space, seed=params_seed)
    sweeps = sobol.gen(n=n_sweeps).arms
    if skip_sweeps is not None:
        sweeps = sweeps[skip_sweeps:]

    for i, sweep in enumerate(sweeps):
        train_catboost(
            max_trees=4096,
            experiment_name="%s_%d_%s" % (dataset_name, i, time_suffix),
            dataset=dataset,
            device="GPU" if use_gpu else "CPU",
            report_frequency=100,
            output_dir=output_dir,
            model_seed=model_seed,
            verbose=verbose,
            **sweep.parameters,
        )


def tune_fcn(
    n_sweeps,
    time_suffix,
    dataset_name,
    dataset,
    use_gpu,
    output_dir,
    model_seed,
    params_seed,
    verbose,
    skip_sweeps=None,
):
    n_head_units = RangeParameter(name="n_head_units", parameter_type=ParameterType.INT, lower=8, upper=10)
    n_tail_units = RangeParameter(name="n_tail_units", parameter_type=ParameterType.INT, lower=7, upper=9)
    order_constraint = OrderConstraint(
        lower_parameter=n_tail_units,
        upper_parameter=n_head_units,
    )

    search_space = SearchSpace(
        parameters=[
            n_head_units,
            n_tail_units,
            RangeParameter(name="n_head_layers", parameter_type=ParameterType.INT, lower=1, upper=2),
            RangeParameter(name="n_tail_layers", parameter_type=ParameterType.INT, lower=1, upper=4),
            ChoiceParameter(name="dropout", parameter_type=ParameterType.FLOAT, values=[0.0, 0.1, 0.2, 0.3]),
            RangeParameter(name="learning_rate", parameter_type=ParameterType.FLOAT, lower=1e-4, upper=1e-2, log_scale=True),
        ],
        parameter_constraints=[order_constraint]
    )


    sobol = get_sobol(search_space=search_space, seed=params_seed)
    sweeps = sobol.gen(n=n_sweeps).arms
    if skip_sweeps is not None:
        sweeps = sweeps[skip_sweeps:]

    for i, sweep in enumerate(sweeps):
        train_fcn(
            experiment_name="%s_%d_%s" % (dataset_name, i, time_suffix),
            dataset=dataset,
            batch_size=1024,
            device="cuda" if use_gpu else "cpu",
            report_frequency=100,
            epochs=float("inf"),
            output_dir=output_dir,
            model_seed=model_seed,
            verbose=verbose,
            **sweep.parameters,
        )


if __name__ == "__main__":
    args = get_args()

    if args.cuda and torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False

    dataset = get_dataset(
        dataset=args.dataset,
        standardize=args.standardize,
        seed=args.dataset_seed,
        quantile_noise=1e-3 if args.model_name == "catboost" else 0
    )

    time_suffix = '{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}'.format(*time.gmtime()[:5])

    if args.model_name == "catboost":
        tune_catboost(
            n_sweeps=args.n_sweeps,
            time_suffix=time_suffix,
            dataset_name=args.dataset,
            dataset=dataset,
            use_gpu=use_gpu,
            output_dir=args.output_dir,
            model_seed=args.model_seed,
            params_seed=args.params_seed,
            verbose=args.verbose,
            skip_sweeps=args.skip_sweeps
        )
    elif args.model_name == "fcn":
        tune_fcn(
            n_sweeps=args.n_sweeps,
            time_suffix=time_suffix,
            dataset_name=args.dataset,
            dataset=dataset,
            use_gpu=use_gpu,
            output_dir=args.output_dir,
            model_seed=args.model_seed,
            params_seed=args.params_seed,
            verbose=args.verbose,
            skip_sweeps=args.skip_sweeps
        )
    else:
        raise NotImplementedError
