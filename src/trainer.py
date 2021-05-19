import os
import json
import numpy as np
import shutil
import random
from sklearn.metrics import roc_auc_score, log_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

from catboost import CatBoostRegressor, CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier

import node.lib
from node.lib import check_numpy, to_one_hot, Lambda

from qhoptim.pyt import QHAdam

from utils import process_in_chunks, get_attention_function
from model import TabTransformer, get_random_masks, get_tree_masks, get_window_masks


class Trainer(node.lib.Trainer):
    def __init__(self, *args, mask_fraction=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_fraction = mask_fraction

    def evaluate_classification_error(self, X_test, y_test, device, batch_size=4096):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            logits = process_in_chunks(self.model, X_test, batch_size=batch_size)
            logits = check_numpy(logits)
            error_rate = (y_test != np.argmax(logits, axis=1)).mean()
        return error_rate

    def evaluate_mse(self, X_test, y_test, device, batch_size=4096):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            prediction = process_in_chunks(self.model, X_test, batch_size=batch_size)
            prediction = check_numpy(prediction)
            error_rate = ((y_test - prediction) ** 2).mean()
        return error_rate

    def evaluate_mae(self, X_test, y_test, device, batch_size=4096):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            prediction = process_in_chunks(self.model, X_test, batch_size=batch_size)
            prediction = check_numpy(prediction)
            error_rate = np.abs(y_test - prediction).mean()
        return error_rate

    def evaluate_auc(self, X_test, y_test, device, batch_size=512):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            logits = F.softmax(process_in_chunks(self.model, X_test, batch_size=batch_size), dim=1)
            logits = check_numpy(logits)
            y_test = torch.from_numpy(y_test)
            auc = roc_auc_score(check_numpy(to_one_hot(y_test)), logits)
        return auc

    def evaluate_logloss(self, X_test, y_test, device, batch_size=512):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        self.model.train(False)
        with torch.no_grad():
            logits = F.softmax(process_in_chunks(self.model, X_test, batch_size=batch_size), dim=1)
            logits = check_numpy(logits)
            y_test = torch.from_numpy(y_test)
            logloss = log_loss(check_numpy(to_one_hot(y_test)), logits)
        return logloss

    def pretrain_loss(self, X):
        mask = (torch.rand_like(X) < self.mask_fraction).float()
        mask_unk = (torch.rand_like(X) < 0.9).float()
        pretrain_mask = (mask * mask_unk).unsqueeze(-1)
        loss = (self.model(X, pretrain_mask=pretrain_mask) - X) ** 2
        loss = (loss * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-16)
        return loss

    def evaluate_pretrain(self, X_test, device, batch_size=512):
        X_test = torch.as_tensor(X_test, device=device)
        self.model.train(False)
        with torch.no_grad():
            loss = process_in_chunks(self.pretrain_loss, X_test, batch_size=batch_size).cpu().detach().numpy()
        return float(loss.mean())

    def pretrain_on_batch(self, *batch, device):
        x_batch, y_batch = batch
        x_batch = torch.as_tensor(x_batch, device=device)
        y_batch = torch.as_tensor(y_batch, device=device)

        self.model.train()
        self.opt.zero_grad()
        loss = self.pretrain_loss(x_batch).mean()
        loss.backward()
        self.opt.step()
        self.step += 1
        self.writer.add_scalar('train loss', loss.item(), self.step)

        return {'loss': loss}

    def forward(self):
        raise NotImplementedError


def train(
    trainer,
    data,
    batch_size,
    device,
    report_frequency,
    dataset_task,
    early_stopping_rounds=1000,
    targets_std=1,
    n_classes=None,
    verbose=True,
    epochs=float("inf"),
    pretrain=False
):

    best_metric = float("inf")
    best_step_metric = 0

    for batch in node.lib.iterate_minibatches(data.X_train, data.y_train, batch_size=batch_size,
                                                shuffle=True, epochs=epochs):
        if pretrain:
            trainer.pretrain_on_batch(*batch, device=device)
        else:
            trainer.train_on_batch(*batch, device=device)

        if trainer.step % report_frequency == 0:
            trainer.save_checkpoint()
            trainer.average_checkpoints(out_tag='avg')
            trainer.load_checkpoint(tag='avg')

            if pretrain:
                metric = trainer.evaluate_pretrain(data.X_valid, device, batch_size=1024)
            elif dataset_task == "regression":
                metric = trainer.evaluate_mse(data.X_valid, data.y_valid, device, batch_size=1024)
                metric *= targets_std ** 2
            elif dataset_task == "classification":
                metric = trainer.evaluate_classification_error(data.X_valid, data.y_valid, device, batch_size=1024)

            trainer.writer.add_scalar('val metric', metric, trainer.step)

            if metric < best_metric:
                best_metric = metric
                best_step_metric = trainer.step
                trainer.save_checkpoint(tag='best')

            trainer.load_checkpoint()  # last
            trainer.remove_old_temp_checkpoints()

            if verbose:
                print("Step {}. Val Metric: {:.5f}".format(trainer.step, metric))

        if trainer.step > best_step_metric + early_stopping_rounds:
            if verbose:
                print('BREAK. There is no improvment for {} steps'.format(early_stopping_rounds))
                print("Best step: ", best_step_metric)
                print("Best Val Metric: %0.5f" % (best_metric))
            break

    if verbose:
        print("Computing final metrics...")

    results = {}
    trainer.load_checkpoint(tag='best')
    for name, (X, y) in {"test": (data.X_test, data.y_test), "valid": (data.X_valid, data.y_valid)}.items():
        if pretrain:
            results[name] = {
                "pretrain": trainer.evaluate_pretrain(X, device, batch_size=512)
            }
        elif dataset_task == "regression":
            results[name] = {
                "mse": trainer.evaluate_mse(X, y, device, batch_size=512) * targets_std ** 2,
                "mae": trainer.evaluate_mae(X, y, device, batch_size=512) * targets_std,
            }
        elif dataset_task == "classification":
            results[name] = {
                "clf-error": trainer.evaluate_classification_error(X, y, device, batch_size=512),
                "logloss": trainer.evaluate_logloss(X, y, device, batch_size=512),
            }
            if n_classes == 2:
                results[name]["auc-roc"] = trainer.evaluate_auc(X, y, device, batch_size=512)
        else:
            raise ValueError("Unknown dataset task")

    eval_path = os.path.join(trainer.experiment_path, "eval.json")
    with open(eval_path, "w") as _out:
        json.dump(results, _out)

    if verbose:
        print("Cleaning up...")

    trainer.remove_old_temp_checkpoints(number_ckpts_to_keep=0)
    avg_path = os.path.join(trainer.experiment_path, "checkpoint_avg.pth")
    os.remove(avg_path)

    if verbose:
        print("Finished!\n")

    return results


def get_default_optimizer_params():
    return {
        "nus": (0.7, 1.0),
        "betas": (0.95, 0.998)
    }


def parse_attention_functions(attention_function):
    if ":" in attention_function:
        return attention_function.split(":", 1)
    return attention_function, attention_function


def train_tab_transformer(
    n_tokens,
    n_transformers,
    d_model,
    attention_function,
    n_heads,
    experiment_name,
    dataset,
    batch_size,
    device,
    report_frequency,
    mask,
    dim_ff_factor=2,
    dropout=0,
    epochs=None,
    output_dir=None,
    model_seed=42,
    verbose=True,
    pretrain=False,
    mask_fraction=None,
    n_to_freeze=None,
):
    torch.manual_seed(model_seed)
    random.seed(model_seed)
    np.random.seed(model_seed)

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
        dim_feedforward=dim_ff_factor * d_model,
        dim_output=dataset.dim_output,
        attention_kwargs=attention_kwargs,
        agg_attention_kwargs=agg_attention_kwargs,
        masks=masks,
        dropout=dropout,
        with_pretrain_stage=pretrain
    ).to(device)

    if pretrain:
        trainer = Trainer(
            model=model,
            loss_function=None,
            experiment_name="pretrain_" + experiment_name,
            warm_start=False,
            Optimizer=QHAdam,
            optimizer_params=get_default_optimizer_params(),
            verbose=verbose,
            n_last_checkpoints=5,
            mask_fraction=mask_fraction
        )
        trainer.model.pretrain_on()

        train(
            trainer=trainer,
            data=dataset.data,
            batch_size=batch_size,
            device=device,
            report_frequency=report_frequency,
            dataset_task=dataset.dataset_task,
            epochs=epochs or float("inf"),
            n_classes=dataset.n_classes,
            targets_std=dataset.target_std,
            verbose=verbose,
            pretrain=True,
        )
        trainer.model.pretrain_off()

        if n_to_freeze is not None:
            trainer.model.freeze_transformers(n_to_freeze)

    trainer = Trainer(
        model=model,
        loss_function=F.mse_loss if dataset.dataset_task == "regression" else F.cross_entropy,
        experiment_name=experiment_name,
        warm_start=False,
        Optimizer=QHAdam,
        optimizer_params=get_default_optimizer_params(),
        verbose=verbose,
        n_last_checkpoints=5
    )

    metrics = train(
        trainer=trainer,
        data=dataset.data,
        batch_size=batch_size,
        device=device,
        report_frequency=report_frequency,
        dataset_task=dataset.dataset_task,
        epochs=epochs or float("inf"),
        n_classes=dataset.n_classes,
        targets_std=dataset.target_std,
        verbose=verbose,
    )

    params=dict(
        n_tokens=n_tokens,
        d_model=d_model,
        n_transformers=n_transformers,
        dim_ff_factor=dim_ff_factor,
        mask=mask,
        dropout=dropout,
        attention_function=attention_function,
        n_heads=n_heads,
    )

    params_path = os.path.join(trainer.experiment_path, "params.json")
    with open(params_path, "w") as _out:
        json.dump(params, _out)

    if output_dir is not None:
        shutil.move(trainer.experiment_path, output_dir)

    return metrics


def train_catboost(
    max_trees,
    learning_rate,
    l2_leaf_reg,
    leaf_estimation_iterations,
    random_strength,
    subsample,
    experiment_name,
    dataset,
    device,
    report_frequency,
    output_dir=None,
    model_seed=42,
    verbose=True,
):
    if dataset.dataset_task == "regression":
        estimator = CatBoostRegressor
    elif dataset.dataset_task == "classification":
        estimator = CatBoostClassifier
    else:
        raise ValueError("Unknown dataset task")

    model = estimator(
        iterations=max_trees,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        leaf_estimation_iterations=leaf_estimation_iterations,
        random_strength=random_strength,
        subsample=subsample,
        task_type=device,
        random_seed=model_seed,
    )

    data = dataset.data
    model.fit(
        X=data.X_train,
        y=data.y_train,
        verbose=verbose,
        metric_period=report_frequency,
    )

    results = {}
    for name, (X, y) in {"test": (data.X_test, data.y_test), "valid": (data.X_valid, data.y_valid)}.items():
        if dataset.dataset_task == "regression":
            y_pred = model.predict(X)
            results[name] = {
                "mse": np.mean((y - y_pred) ** 2) * dataset.target_std ** 2,
                "mae": np.mean(np.abs(y - y_pred)) * dataset.target_std,
            }
        elif dataset.dataset_task == "classification":
            y_pred = model.predict_proba(X)
            one_hot = np.zeros((y.shape[0], dataset.n_classes))
            one_hot[np.arange(one_hot.shape[0]), y] = 1
            results[name] = {
                "clf-error": np.mean(np.argmax(y_pred, -1) != y),
                "logloss": log_loss(one_hot, y_pred)
            }
            if dataset.n_classes == 2:
                results[name]["auc-roc"] = roc_auc_score(one_hot, y_pred)

    if output_dir is not None:
        out_path = os.path.join(output_dir, experiment_name)
        os.mkdir(out_path)

        eval_path = os.path.join(out_path, "eval.json")
        with open(eval_path, "w") as _out:
            json.dump(results, _out)

        model_path = os.path.join(out_path, "model.cb")
        model.save_model(model_path)

        params_path = os.path.join(out_path, "params.json")
        params = dict(
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            leaf_estimation_iterations=leaf_estimation_iterations,
            random_strength=random_strength,
            subsample=subsample,
        )
        with open(params_path, "w") as _out:
            json.dump(params, _out)

    if verbose:
        print("Finished!")

    return results


def train_fcn(
    n_head_layers,
    n_head_units,
    n_tail_layers,
    n_tail_units,
    dropout,
    learning_rate,
    experiment_name,
    dataset,
    batch_size,
    device,
    report_frequency,
    epochs=None,
    output_dir=None,
    model_seed=42,
    verbose=True,
):
    torch.manual_seed(model_seed)
    random.seed(model_seed)
    np.random.seed(model_seed)

    model = []
    units = (
        [dataset.data.X_train.shape[1]] +
        [2 ** n_head_units] * n_head_layers +
        [2 ** n_tail_units] * n_tail_layers
    )
    for i in range(1, len(units)):
        model.extend([
            nn.Linear(units[i - 1], units[i]),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
    model.extend([
        nn.Linear(units[-1], dataset.dim_output),
        Lambda(lambda x: x.squeeze(1))
    ])
    model = nn.Sequential(*model).to(device)

    optimizer_params = get_default_optimizer_params()
    optimizer_params["lr"] = learning_rate
    trainer = Trainer(
        model=model,
        loss_function=F.mse_loss if dataset.dataset_task == "regression" else F.cross_entropy,
        experiment_name=experiment_name,
        warm_start=False,
        Optimizer=QHAdam,
        optimizer_params=optimizer_params,
        verbose=verbose,
        n_last_checkpoints=5
    )

    metrics = train(
        trainer=trainer,
        data=dataset.data,
        batch_size=batch_size,
        device=device,
        report_frequency=report_frequency,
        dataset_task=dataset.dataset_task,
        epochs=epochs or float("inf"),
        n_classes=dataset.n_classes,
        targets_std=dataset.target_std,
        verbose=verbose,
    )

    params_path = os.path.join(trainer.experiment_path, "params.json")
    params = dict(
        n_head_layers=n_head_layers,
        n_head_units=n_head_units,
        n_tail_layers=n_tail_layers,
        n_tail_units=n_tail_units,
        dropout=dropout,
        learning_rate=learning_rate,
    )
    with open(params_path, "w") as _out:
        json.dump(params, _out)

    if output_dir is not None:
        shutil.move(trainer.experiment_path, output_dir)

    return metrics


def train_node(
    num_layers,
    total_tree_count,
    tree_depth,
    tree_output_dim,
    experiment_name,
    dataset,
    batch_size,
    device,
    report_frequency,
    epochs=None,
    output_dir=None,
    model_seed=42,
    verbose=True,
):
    torch.manual_seed(model_seed)
    random.seed(model_seed)
    np.random.seed(model_seed)

    if dataset.dataset_task == "regression":
        averaging = node.lib.Lambda(lambda x: x[..., 0].mean(dim=-1))
    else:
        averaging = node.lib.Lambda(lambda x: x[..., :dataset.n_classes].mean(dim=-2))

    model = nn.Sequential(
        node.lib.DenseBlock(
            dataset.n_features,
            total_tree_count,
            num_layers=num_layers,
            tree_dim=tree_output_dim,
            depth=tree_depth,
            flatten_output=False,
            choice_function=node.lib.entmax15,
            bin_function=node.lib.entmoid15
        ),
        averaging,
    ).to(device)

    # Data-aware initialization
    with torch.no_grad():
        x = model(torch.as_tensor(dataset.data.X_train[:2000], device=device))

    trainer = Trainer(
        model=model,
        loss_function=F.mse_loss if dataset.dataset_task == "regression" else F.cross_entropy,
        experiment_name=experiment_name,
        warm_start=False,
        Optimizer=QHAdam,
        optimizer_params=get_default_optimizer_params(),
        verbose=verbose,
        n_last_checkpoints=5
    )

    metrics = train(
        trainer=trainer,
        data=dataset.data,
        batch_size=batch_size,
        device=device,
        report_frequency=report_frequency,
        dataset_task=dataset.dataset_task,
        epochs=epochs or float("inf"),
        n_classes=dataset.n_classes,
        targets_std=dataset.target_std,
        verbose=verbose,
    )

    params=dict(
        num_layers=num_layers,
        total_tree_count=total_tree_count,
        tree_depth=tree_depth,
        tree_output_dim=tree_output_dim,
    )

    params_path = os.path.join(trainer.experiment_path, "params.json")
    with open(params_path, "w") as _out:
        json.dump(params, _out)

    if output_dir is not None:
        shutil.move(trainer.experiment_path, output_dir)

    return metrics


def train_tabnet(
    n_d,
    n_steps,
    gamma,
    n_independent,
    n_shared,
    lambda_sparse,
    learning_rate,
    mask_type,
    experiment_name,
    dataset,
    batch_size,
    device,
    epochs,
    patience,
    output_dir=None,
    model_seed=42,
    verbose=1,
):
    if dataset.dataset_task == "regression":
        estimator = TabNetRegressor
    else:
        estimator = TabNetClassifier

    model = estimator(
        n_d=n_d,
        n_a=n_d,
        n_steps=n_steps,
        gamma=gamma,
        n_independent=n_independent,
        n_shared=n_shared,
        lambda_sparse=lambda_sparse,
        mask_type=mask_type,
        seed=model_seed,
        optimizer_params={"lr": learning_rate},
        verbose=verbose,
        device_name=device,
    )

    if dataset.dataset_task == "regression":
        y_transform = lambda y: y.reshape(-1, 1)
    else:
        y_transform = lambda y: y

    data = dataset.data
    model.fit(
        X_train=data.X_train,
        y_train=y_transform(data.y_train),
        eval_set=[(data.X_valid, y_transform(data.y_valid))],
        eval_metric=["mse" if dataset.dataset_task == "regression" else "accuracy"],
        max_epochs=epochs,
        patience=patience,
        batch_size=batch_size,
    )

    results = {}
    for name, (X, y) in {"test": (data.X_test, data.y_test), "valid": (data.X_valid, data.y_valid)}.items():
        if dataset.dataset_task == "regression":
            y_pred = model.predict(X).reshape(-1)
            results[name] = {
                "mse": np.mean((y - y_pred) ** 2) * dataset.target_std ** 2,
                "mae": np.mean(np.abs(y - y_pred)) * dataset.target_std,
            }
        elif dataset.dataset_task == "classification":
            y_pred = model.predict_proba(X)
            one_hot = np.zeros((y.shape[0], dataset.n_classes))
            one_hot[np.arange(one_hot.shape[0]), y] = 1
            results[name] = {
                "clf-error": np.mean(np.argmax(y_pred, -1) != y),
                "logloss": log_loss(one_hot, y_pred)
            }
            if dataset.n_classes == 2:
                results[name]["auc-roc"] = roc_auc_score(one_hot, y_pred)

    if output_dir is not None:
        out_path = os.path.join(output_dir, experiment_name)
        os.mkdir(out_path)

        eval_path = os.path.join(out_path, "eval.json")
        with open(eval_path, "w") as _out:
            json.dump(results, _out)

        model_path = os.path.join(out_path, "model.tabnet")
        model.save_model(model_path)

        params_path = os.path.join(out_path, "params.json")
        params = dict(
            n_d=n_d,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            lambda_sparse=lambda_sparse,
            learning_rate=learning_rate,
            mask_type=mask_type,
        )
        with open(params_path, "w") as _out:
            json.dump(params, _out)

    return results