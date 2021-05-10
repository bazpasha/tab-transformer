from collections import namedtuple
import numpy as np

import node.lib


DatasetInfo = namedtuple("DatasetInfo", ("data", "target_std", "n_classes", "dataset_task", "dim_output", "n_features"))


def get_dataset_task(dataset):
    if dataset.dataset in ["YEAR", "MICROSOFT", "YAHOO"]:
        return "regression"
    elif dataset.dataset in ["EPSILON", "CLICK", "HIGGS"]:
        return "classification"
    else:
        raise ValueError("Unknown dataset")


def get_dataset(dataset, seed, standardize, quantile_noise):
    data = node.lib.Dataset(
        dataset=dataset,
        random_state=seed,
        quantile_transform=True,
        quantile_noise=quantile_noise
    )

    dataset_task = get_dataset_task(data)

    std = 1
    if standardize:
        if dataset_task != "regression":
            raise ValueError("Can't standardize targets for non-regression dataset")
        mu, std = data.y_train.mean(), data.y_train.std()
        normalize = lambda x: ((x - mu) / std).astype(np.float32)
        data.y_train, data.y_valid, data.y_test = map(normalize, [data.y_train, data.y_valid, data.y_test])

    n_classes = None
    if dataset_task == "classification":
        n_classes = len(set(data.y_train))

    dim_output = 1 if dataset_task == "regression" else n_classes

    return DatasetInfo(
        data=data,
        dataset_task=dataset_task,
        dim_output=dim_output,
        n_classes=n_classes,
        target_std=std,
        n_features=data.X_train.shape[1],
    )
