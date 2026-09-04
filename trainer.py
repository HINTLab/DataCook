    """
    Adapted from https://github.com/HanxunH/Unlearnable-Examples
    Implements a trainer extended functionality.
    """
import time
import models
import torch
import util
import torch.nn.functional as F

from models.generator import ResnetGenerator
from models.generator3d import ResnetGenerator3D

import os 

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class Trainer():
    def __init__(self, criterion, data_loader, logger, config, train_data_type, global_step=0,
                 target='train_dataset'):
        self.criterion = criterion
        self.data_loader = data_loader
        self.logger = logger
        self.config = config
        self.log_frequency = config.log_frequency if config.log_frequency is not None else 100
        self.loss_meters = util.AverageMeter()
        self.loss_meters_gnet = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.global_step = global_step
        self.target = target
        self.train_data_type = train_data_type
        print(self.target)

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.loss_meters_gnet = util.AverageMeter()

    def train(self, epoch, model, criterion, optimizer, random_noise=None):
        model.train()
        for i, (images, labels,_) in enumerate(self.data_loader[self.target]):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
  
            if random_noise is not None:
                random_noise = random_noise.detach().to(device)
                for i in range(len(labels)):
                    class_index = labels[i].item()
                    images[i] += random_noise[class_index].clone()
                    images[i] = torch.clamp(images[i], 0, 1)
            start = time.time()
            log_payload = self.train_batch(images, labels, model, optimizer)
            end = time.time()
            time_used = end - start
            if self.global_step % self.log_frequency == 0:
                display = util.log_display(epoch=epoch,
                                           global_step=self.global_step,
                                           time_elapse=time_used,
                                           **log_payload)
                self.logger.info(display)
            self.global_step += 1
        return self.global_step

    def train_model(self, epoch, model, criterion, optimizer, random_noise=None):
        model.train()
        for i, (images, labels) in enumerate(self.data_loader[self.target]):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            if self.train_data_type == 'PathMNIST' or self.train_data_type == 'DermaMNIST' or self.train_data_type == 'OCTMNIST' or self.train_data_type == 'PneumoniaMNIST' or self.train_data_type == 'RetinaMNIST' or  self.train_data_type == 'BreastMNIST' or self.train_data_type == 'BloodMNIST' or self.train_data_type == 'TissueMNIST' or self.train_data_type == 'OrganAMNIST' or self.train_data_type == 'OrganCMNIST' or self.train_data_type == 'OrganSMNIST':
                labels = torch.squeeze(labels, 1).long().to(device)
            if random_noise is not None:
                random_noise = random_noise.detach().to(device)
                for i in range(len(labels)):
                    class_index = labels[i].item()
                    images[i] += random_noise[class_index].clone()
                    images[i] = torch.clamp(images[i], 0, 1)
            start = time.time()
            log_payload = self.train_batch(images, labels, model, optimizer)
            end = time.time()
            time_used = end - start
            if self.global_step % self.log_frequency == 0:
                display = util.log_display(epoch=epoch,
                                           global_step=self.global_step,
                                           time_elapse=time_used,
                                           **log_payload)
                self.logger.info(display)
            self.global_step += 1
        return self.global_step

    def train_3d_model(self, epoch, model, criterion, optimizer, random_noise=None):
        model.train()
        for i, (images, labels) in enumerate(self.data_loader[self.target]):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            labels = torch.squeeze(labels, 1).long().to(device)
            if random_noise is not None:
                random_noise = random_noise.detach().to(device)
                for i in range(len(labels)):
                    class_index = labels[i].item()
                    images[i] += random_noise[class_index].clone()
                    images[i] = torch.clamp(images[i], 0, 1)
            start = time.time()
            log_payload = self.train_batch(images, labels, model, optimizer)
            end = time.time()
            time_used = end - start
            if self.global_step % self.log_frequency == 0:
                display = util.log_display(epoch=epoch,
                                           global_step=self.global_step,
                                           time_elapse=time_used,
                                           **log_payload)
                self.logger.info(display)
            self.global_step += 1
        return self.global_step

    def gnet_train_3d(self, num_clusters, cluster, g_net_num_epoch, g_net_lr, model, epsilon, logger, perturbation_path, data_loader, target_offset=1):
        model = model.eval()

        for cluster_idx in range(num_clusters):
            if self.train_data_type in ['OrganMNIST3D', 'NoduleMNIST3D', 'FractureMNIST3D','AdrenalMNIST3D','VesselMNIST3D','SynapseMNIST3D']:
                logger.info("Process grey image")
                noise = torch.zeros((1, 1, 28, 28, 28))
                noise.uniform_(0, 1)
                # noise = noise.repeat(1, 3, 1, 1, 1)
                noise = noise.to(device)
            else:
                logger.info("Process RGB image")
                noise = torch.zeros((1, 1, 28, 28, 28))
                noise.uniform_(0, 1)
                noise = noise.to(device)

            g_net = ResnetGenerator3D(1, 1, 64, norm_type='batch', act_type='relu')
            g_net.to(device)
            optimizer = torch.optim.Adam(g_net.parameters(), lr=g_net_lr, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, g_net_num_epoch * len(data_loader["train_dataset"]), eta_min=1e-6)
            criterion = torch.nn.KLDivLoss(reduction='batchmean')
            
            features = {}
            def hook(layer, inp, out):
                features['feat'] = inp[0]
            model.fc.register_forward_hook(hook)

            for epoch in range(0, g_net_num_epoch):
                logger.info("")
                logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
                
                g_net.train()
                for i, (images, labels) in enumerate(self.data_loader[self.target]):
                    
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    delta_im = g_net(noise).repeat(images.shape[0], 1, 1, 1, 1)
                    delta_im = torch.clamp(delta_im, 0, epsilon)
                    images_adv = torch.clamp(images + delta_im, 0, 1)
                    target_labels = (torch.ones(len(images)).long() * cluster_idx + target_offset) % num_clusters
                    target_labels = target_labels.to(device)
                    anchors = torch.stack([cluster['centers'][i] for i in target_labels], dim=0).to(device)
                    start = time.time()

                    model(images_adv)
                    loss = criterion(features['feat'].log_softmax(dim=-1), anchors.softmax(dim=-1)) 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    self.loss_meters.update(loss.item(), labels.shape[0])
                    payload = {"Class_idx": cluster_idx,
                        "loss": loss,
                        "loss_avg": self.loss_meters.avg,
                        "lr": optimizer.param_groups[0]['lr']}
                    end = time.time()
                    time_used = end - start
                    if self.global_step % self.log_frequency == 0:
                        display = util.log_display(epoch=epoch,
                                                global_step=self.global_step,
                                                time_elapse=time_used,
                                                **payload)
                        self.logger.info(display)
                    self.global_step += 1 
                with torch.no_grad():
                    perturbation = g_net(noise)
                torch.save({'state_dict': g_net.state_dict(), 'init_noise': noise, 'perturbation': perturbation}, os.path.join(perturbation_path, f'perturbation_{cluster_idx}.pth'))
                self._reset_stats()
        return self.global_step, perturbation, g_net

    def gnet_train(self, num_clusters, cluster, g_net_num_epoch, g_net_lr, model, epsilon, logger, perturbation_path, data_loader, target_offset=1):
        model = model.eval()

        for cluster_idx in range(num_clusters):
            if self.train_data_type in ['OrganAMNIST', 'OrganCMNIST', 'OrganSMNIST', 'OCTMNIST', 'PneumoniaMNIST', 'BreastMNIST', 'TissueMNIST']:
                logger.info("Process grey image")
                noise = torch.zeros((1, 1, 28, 28))
                noise.uniform_(0, 1)
                noise = noise.repeat(1, 3, 1, 1)
                noise = noise.to(device)
            else:
                logger.info("Process RGB image")
                noise = torch.zeros((1, 3, 28, 28))
                noise.uniform_(0, 1)
                noise = noise.to(device)

            g_net = ResnetGenerator(3, 3, 64, norm_type='batch', act_type='relu')
            g_net.to(device)
            optimizer = torch.optim.Adam(g_net.parameters(), lr=g_net_lr, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, g_net_num_epoch * len(data_loader["train_dataset"]), eta_min=1e-6)
            criterion = torch.nn.KLDivLoss(reduction='batchmean')
            
            features = {}
            def hook(layer, inp, out):
                features['feat'] = inp[0]
            model.fc.register_forward_hook(hook)

            for epoch in range(0, g_net_num_epoch):
                logger.info("")
                logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
                
                g_net.train()
                for i, (images, labels) in enumerate(self.data_loader[self.target]):
                    
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                    delta_im = g_net(noise).repeat(images.shape[0], 1, 1, 1)

                    delta_im = torch.clamp(delta_im, 0, epsilon)
                    images_adv = torch.clamp(images + delta_im, 0, 1)
                    target_labels = (torch.ones(len(images)).long() * cluster_idx + target_offset) % num_clusters
                    target_labels = target_labels.to(device)
                    anchors = torch.stack([cluster['centers'][i] for i in target_labels], dim=0).to(device)
                    start = time.time()

                    model(images_adv)
                    loss = criterion(features['feat'].log_softmax(dim=-1), anchors.softmax(dim=-1)) 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    self.loss_meters.update(loss.item(), labels.shape[0])
                    payload = {"Class_idx": cluster_idx,
                        "loss": loss,
                        "loss_avg": self.loss_meters.avg,
                        "lr": optimizer.param_groups[0]['lr']}
                    end = time.time()
                    time_used = end - start
                    if self.global_step % self.log_frequency == 0:
                        display = util.log_display(epoch=epoch,
                                                global_step=self.global_step,
                                                time_elapse=time_used,
                                                **payload)
                        self.logger.info(display)
                    self.global_step += 1 
                with torch.no_grad():
                    perturbation = g_net(noise)
                torch.save({'state_dict': g_net.state_dict(), 'init_noise': noise, 'perturbation': perturbation}, os.path.join(perturbation_path, f'perturbation_{cluster_idx}.pth'))
                self._reset_stats()
        return self.global_step, perturbation, g_net

    def gnet_trai_224(self, num_clusters, cluster, g_net_num_epoch, g_net_lr, model, epsilon, logger, perturbation_path, data_loader, target_offset=1):
        model = model.eval()

        for cluster_idx in range(num_clusters):
            if self.train_data_type in ['OrganAMNIST', 'OrganCMNIST', 'OrganSMNIST', 'OCTMNIST', 'PneumoniaMNIST', 'BreastMNIST', 'TissueMNIST']:
                logger.info("Process grey image")
                noise = torch.zeros((1, 1, 224, 224))
                noise.uniform_(0, 1)
                noise = noise.repeat(1, 3, 1, 1)
                noise = noise.to(device)
            else:
                logger.info("Process RGB image")
                noise = torch.zeros((1, 3, 224, 224))
                noise.uniform_(0, 1)
                noise = noise.to(device)

            g_net = ResnetGenerator(3, 3, 64, norm_type='batch', act_type='relu')
            g_net.to(device)
            optimizer = torch.optim.Adam(g_net.parameters(), lr=g_net_lr, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, g_net_num_epoch * len(data_loader["train_dataset"]), eta_min=1e-6)
            criterion = torch.nn.KLDivLoss(reduction='batchmean')
            
            features = {}
            def hook(layer, inp, out):
                features['feat'] = inp[0]
            model.fc.register_forward_hook(hook)

            for epoch in range(0, g_net_num_epoch):
                logger.info("")
                logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
                
                g_net.train()
                for i, (images, labels) in enumerate(self.data_loader[self.target]):
                    
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                    delta_im = g_net(noise).repeat(images.shape[0], 1, 1, 1)

                    delta_im = torch.clamp(delta_im, 0, epsilon)
                    images_adv = torch.clamp(images + delta_im, 0, 1)
                    target_labels = (torch.ones(len(images)).long() * cluster_idx + target_offset) % num_clusters
                    target_labels = target_labels.to(device)
                    anchors = torch.stack([cluster['centers'][i] for i in target_labels], dim=0).to(device)
                    start = time.time()

                    model(images_adv)
                    loss = criterion(features['feat'].log_softmax(dim=-1), anchors.softmax(dim=-1)) 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    self.loss_meters.update(loss.item(), labels.shape[0])
                    payload = {"Class_idx": cluster_idx,
                        "loss": loss,
                        "loss_avg": self.loss_meters.avg,
                        "lr": optimizer.param_groups[0]['lr']}
                    end = time.time()
                    time_used = end - start
                    if self.global_step % self.log_frequency == 0:
                        display = util.log_display(epoch=epoch,
                                                global_step=self.global_step,
                                                time_elapse=time_used,
                                                **payload)
                        self.logger.info(display)
                    self.global_step += 1 
                with torch.no_grad():
                    perturbation = g_net(noise)
                torch.save({'state_dict': g_net.state_dict(), 'init_noise': noise, 'perturbation': perturbation}, os.path.join(perturbation_path, f'perturbation_{cluster_idx}.pth'))
                self._reset_stats()
        return self.global_step, perturbation, g_net

    def train_batch(self, images, labels, model, optimizer):
        model.zero_grad()
        optimizer.zero_grad()
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            logits = model(images)
            loss = self.criterion(logits, labels)
        else:
            logits, loss = self.criterion(model, images, labels, optimizer)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        optimizer.step()
        if logits.shape[1] >= 5:
            acc, acc5 = util.accuracy(logits, labels, topk=(1, 5))
            acc, acc5 = acc.item(), acc5.item()
        else:
            acc, = util.accuracy(logits, labels, topk=(1,))
            acc, acc5 = acc.item(), 1
        self.loss_meters.update(loss.item(), labels.shape[0])
        self.acc_meters.update(acc, labels.shape[0])
        self.acc5_meters.update(acc5, labels.shape[0])
        payload = {"acc": acc,
                   "acc_avg": self.acc_meters.avg,
                   "loss": loss,
                   "loss_avg": self.loss_meters.avg,
                   "lr": optimizer.param_groups[0]['lr'],
                   "|gn|": grad_norm}
        return payload
