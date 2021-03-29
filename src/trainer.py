import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

import torch
import torch.nn.functional as F

import node.lib
from node.lib import check_numpy, to_one_hot

from utils import process_in_chunks


class Trainer(node.lib.Trainer):
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
    epochs=float("inf")):

    best_metric = float("inf")
    best_step_metric = 0

    for batch in node.lib.iterate_minibatches(data.X_train, data.y_train, batch_size=batch_size,
                                                shuffle=True, epochs=epochs):
        trainer.train_on_batch(*batch, device=device)

        if trainer.step % report_frequency == 0:
            trainer.save_checkpoint()
            trainer.average_checkpoints(out_tag='avg')
            trainer.load_checkpoint(tag='avg')

            if dataset_task == "regression":
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
        if dataset_task == "regression":
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


def get_default_optimizer_params():
    return {
        "nus": (0.7, 1.0),
        "betas": (0.95, 0.998)
    }
