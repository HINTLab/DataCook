import time
import os
import numpy as np

import torch
import util
from metrics.classification_metrics import compute_classification_metrics

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Evaluator():
    def __init__(self, data_loader, logger, config, data_type):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.task_type = getattr(config, 'task_type', None) or 'multiclass'
        normalized_task = str(self.task_type).replace('-', '').replace('_', '').lower()
        self.is_multilabel = normalized_task == 'multilabel'
        self.criterion = (
            torch.nn.BCEWithLogitsLoss() if self.is_multilabel
            else torch.nn.CrossEntropyLoss()
        )
        self.data_loader = data_loader
        self.logger = logger
        self.log_frequency = config.log_frequency if config.log_frequency is not None else 100
        self.config = config
        self.current_acc = 0
        self.current_acc_top5 = 0
        self.confusion_matrix = torch.zeros(config.num_classes, config.num_classes)
        self.data_type = data_type
        self.y_true = []
        self.y_prob = []
        self.recover_y_true = []
        self.recover_y_prob = []

        self.recover_loss_meters = util.AverageMeter()
        self.recover_acc_meters = util.AverageMeter()
        self.recover_acc5_meters = util.AverageMeter()
        self.recover_confusion_matrix = torch.zeros(self.config.num_classes, self.config.num_classes)
        return

    def classification_metrics(self, recover=False):
        """Return unified metrics while preserving legacy ACC meters."""
        labels = self.recover_y_true if recover else self.y_true
        probabilities = self.recover_y_prob if recover else self.y_prob
        if not labels:
            raise RuntimeError('No predictions collected; call eval/eval_model first.')
        return compute_classification_metrics(
            torch.cat(labels).numpy(), y_prob=torch.cat(probabilities).numpy(),
            task_type=self.task_type)

    def save_predictions(self, output_path, recover=False):
        """Save labels/probabilities in the revision experiment NPZ contract."""
        labels = self.recover_y_true if recover else self.y_true
        probabilities = self.recover_y_prob if recover else self.y_prob
        if not labels:
            raise RuntimeError('No predictions collected; call eval/eval_model first.')
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        np.savez_compressed(
            output_path,
            y_true=torch.cat(labels).numpy(),
            y_prob=torch.cat(probabilities).numpy(),
        )
        if self.logger is not None:
            self.logger.info('Predictions saved at %s', output_path)
        return output_path

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.confusion_matrix = torch.zeros(self.config.num_classes, self.config.num_classes)

        self.recover_loss_meters = util.AverageMeter()
        self.recover_acc_meters = util.AverageMeter()
        self.recover_acc5_meters = util.AverageMeter()
        self.recover_confusion_matrix = torch.zeros(self.config.num_classes, self.config.num_classes)
        self.y_true = []
        self.y_prob = []
        self.recover_y_true = []
        self.recover_y_prob = []
        return

    def eval(self, epoch, model):
        model.eval()
        for i, batch in enumerate(self.data_loader["test_dataset"]):
            images, labels = batch[0], batch[1]
            if labels.ndim > 1 and labels.shape[1] == 1:
                labels = torch.squeeze(labels, 1).long()
            start = time.time()
            log_payload = self.eval_batch(images=images, labels=labels, model=model)
            end = time.time()
            time_used = end - start
        display = util.log_display(epoch=epoch,
                                   global_step=i,
                                   time_elapse=time_used,
                                   **log_payload)
        if self.logger is not None:
            self.logger.info(display)
        model.eval()
        for i, batch in enumerate(self.data_loader["recover_test_dataset"]):
            images, labels = batch[0], batch[1]
            if labels.ndim > 1 and labels.shape[1] == 1:
                labels = torch.squeeze(labels, 1).long()
            
            start = time.time()
            recover_log_payload = self.eval_recover_batch(images=images, labels=labels, model=model)
            end = time.time()
            time_used = end - start
        display = util.log_display(epoch=epoch,
                                   global_step=i,
                                   time_elapse=time_used,
                                   **recover_log_payload)
        if self.logger is not None:
            self.logger.info(display)
        return

    def eval_model(self, epoch, model):
        model.eval()
        for i, batch in enumerate(self.data_loader["test_dataset"]):
            images, labels = batch[0], batch[1]
            if labels.ndim > 1 and labels.shape[1] == 1:
                labels = torch.squeeze(labels, 1).long()

            start = time.time()
            log_payload = self.eval_batch(images=images, labels=labels, model=model)
            end = time.time()
            time_used = end - start
        display = util.log_display(epoch=epoch,
                                   global_step=i,
                                   time_elapse=time_used,
                                   **log_payload)
        if self.logger is not None:
            self.logger.info(display)
        return


    def eval_batch(self, images, labels, model):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with torch.no_grad():
            pred = model(images)
            if self.is_multilabel:
                targets = labels.float()
                if pred.shape != targets.shape:
                    raise ValueError(
                        f'Multi-label target/logit shapes must match; got {targets.shape} and {pred.shape}.'
                    )
                loss = self.criterion(pred, targets)
                probabilities = torch.sigmoid(pred)
                predictions = probabilities >= 0.5
                acc = predictions.eq(targets.bool()).all(dim=1).float().mean()
                acc5 = 1
                self.y_true.append(targets.detach().cpu())
                self.y_prob.append(probabilities.detach().cpu())
            elif pred.shape[1] >= 5:
                loss = self.criterion(pred, labels)
                acc, acc5 = util.accuracy(pred, labels, topk=(1, 5))
                _, preds = torch.max(pred, 1)
                self.y_true.append(labels.detach().cpu())
                self.y_prob.append(torch.softmax(pred, dim=1).detach().cpu())
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    self.confusion_matrix[t.long(), p.long()] += 1
            else:
                loss = self.criterion(pred, labels)
                acc, = util.accuracy(pred, labels, topk=(1,))
                acc5 = 1
                _, preds = torch.max(pred, 1)
                self.y_true.append(labels.detach().cpu())
                self.y_prob.append(torch.softmax(pred, dim=1).detach().cpu())
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    self.confusion_matrix[t.long(), p.long()] += 1

        self.loss_meters.update(loss.item(), n=images.size(0))
        self.acc_meters.update(acc.item(), n=images.size(0))
        # self.acc5_meters.update(acc5.item(), n=images.size(0))
        self.acc5_meters.update(acc5,  n=images.size(0))
        payload = {"acc": acc.item(),
                   "acc_avg": self.acc_meters.avg,
                   "acc5": acc5,
                   "acc5_avg": self.acc5_meters.avg,
                   "loss": loss.item(),
                   "loss_avg": self.loss_meters.avg}
        return payload

    def eval_recover_batch(self, images, labels, model):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.no_grad():
            pred = model(images)
            if self.is_multilabel:
                targets = labels.float()
                if pred.shape != targets.shape:
                    raise ValueError(
                        f'Multi-label target/logit shapes must match; got {targets.shape} and {pred.shape}.'
                    )
                loss = self.criterion(pred, targets)
                probabilities = torch.sigmoid(pred)
                predictions = probabilities >= 0.5
                acc = predictions.eq(targets.bool()).all(dim=1).float().mean()
                acc5 = 1
                self.recover_y_true.append(targets.detach().cpu())
                self.recover_y_prob.append(probabilities.detach().cpu())
            elif pred.shape[1] >= 5:
                loss = self.criterion(pred, labels)
                acc, acc5 = util.accuracy(pred, labels, topk=(1, 5))
                _, preds = torch.max(pred, 1)
                self.recover_y_true.append(labels.detach().cpu())
                self.recover_y_prob.append(torch.softmax(pred, dim=1).detach().cpu())
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    self.recover_confusion_matrix[t.long(), p.long()] += 1
            else:
                loss = self.criterion(pred, labels)
                acc, = util.accuracy(pred, labels, topk=(1,))
                acc5 = 1
                _, preds = torch.max(pred, 1)
                self.recover_y_true.append(labels.detach().cpu())
                self.recover_y_prob.append(torch.softmax(pred, dim=1).detach().cpu())
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    self.recover_confusion_matrix[t.long(), p.long()] += 1

        self.recover_loss_meters.update(loss.item(), n=images.size(0))
        self.recover_acc_meters.update(acc.item(), n=images.size(0))
        # self.recover_acc5_meters.update(acc5.item(), n=images.size(0))
        self.recover_acc5_meters.update(acc5,  n=images.size(0))
        recover_payload = {"acc": acc.item(),
                   "acc_avg": self.recover_acc_meters.avg,
                   "acc5": acc5,
                   "acc5_avg": self.recover_acc5_meters.avg,
                   "loss": loss.item(),
                   "loss_avg": self.recover_loss_meters.avg}
        return recover_payload
