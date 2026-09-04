import time

import models
import torch
import torch.optim as optim
import util
from torch.autograd import Variable

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Evaluator():
    def __init__(self, data_loader, logger, config, data_type):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.data_loader = data_loader
        self.logger = logger
        self.log_frequency = config.log_frequency if config.log_frequency is not None else 100
        self.config = config
        self.current_acc = 0
        self.current_acc_top5 = 0
        self.confusion_matrix = torch.zeros(config.num_classes, config.num_classes)
        self.data_type = data_type

        self.recover_loss_meters = util.AverageMeter()
        self.recover_acc_meters = util.AverageMeter()
        self.recover_acc5_meters = util.AverageMeter()
        self.recover_confusion_matrix = torch.zeros(self.config.num_classes, self.config.num_classes)
        return

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.confusion_matrix = torch.zeros(self.config.num_classes, self.config.num_classes)

        self.recover_loss_meters = util.AverageMeter()
        self.recover_acc_meters = util.AverageMeter()
        self.recover_acc5_meters = util.AverageMeter()
        self.recover_confusion_matrix = torch.zeros(self.config.num_classes, self.config.num_classes)
        return

    def eval(self, epoch, model):
        model.eval()
        for i, (images, labels) in enumerate(self.data_loader["test_dataset"]):
            if self.data_type == 'PathMNIST' or self.data_type == 'DermaMNIST' or self.data_type == 'OCTMNIST' or\
               self.data_type == 'PneumoniaMNIST' or self.data_type == 'RetinaMNIST' or  self.data_type == 'BreastMNIST' or \
               self.data_type == 'BloodMNIST' or self.data_type == 'TissueMNIST' or self.data_type == 'OrganAMNIST' or \
               self.data_type == 'OrganCMNIST' or self.data_type == 'OrganSMNIST' or self.data_type == 'OrganMNIST3D'or \
               self.data_type == 'NoduleMNIST3D' or self.data_type =='FractureMNIST3D' or self.data_type == 'AdrenalMNIST3D'or \
               self.data_type =='VesselMNIST3D' or self.data_type =='SynapseMNIST3D':
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
        for i, (images, labels, _) in enumerate(self.data_loader["recover_test_dataset"]):
            
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
        for i, (images, labels) in enumerate(self.data_loader["test_dataset"]):
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
            loss = self.criterion(pred, labels)
            if pred.shape[1] >= 5:
                acc, acc5 = util.accuracy(pred, labels, topk=(1, 5))
            else:
                acc, = util.accuracy(pred, labels, topk=(1,))
                acc5 = 1
            # acc, acc5 = util.accuracy(pred, labels, topk=(1, 5))
            _, preds = torch.max(pred, 1)
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
            loss = self.criterion(pred, labels)
            if pred.shape[1] >= 5:
                acc, acc5 = util.accuracy(pred, labels, topk=(1, 5))
            else:
                acc, = util.accuracy(pred, labels, topk=(1,))
                acc5 = 1
            # acc, acc5 = util.accuracy(pred, labels, topk=(1, 5))
            _, preds = torch.max(pred, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                self.recover_confusion_matrix[t.long(), p.long()] += 1

        self.recover_loss_meters.update(loss.item(), n=images.size(0))
        self.recover_acc_meters.update(acc.item(), n=images.size(0))
        # self.recover_acc5_meters.update(acc5.item(), n=images.size(0))
        self.acc5_meters.update(acc5,  n=images.size(0))
        recover_payload = {"acc": acc.item(),
                   "acc_avg": self.recover_acc_meters.avg,
                   "acc5": acc5,
                   "acc5_avg": self.recover_acc5_meters.avg,
                   "loss": loss.item(),
                   "loss_avg": self.recover_loss_meters.avg}
        return recover_payload
