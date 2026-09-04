import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.datasets import make_classification
import medmnist
from medmnist import INFO, Evaluator
from pathlib import Path
import os

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class PerturbationTool():
    def __init__(self, seed=0, epsilon=0.03137254901, num_steps=20, step_size=0.00784313725, imgsize=28, patchsize=7):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.seed = seed
        self.imgsize = imgsize
        self.patchsize = patchsize
        np.random.seed(seed)

    def random_noise(self, noise_shape=[10, 3, 32, 32]):
        random_noise = torch.FloatTensor(*noise_shape).uniform_(-self.epsilon, self.epsilon).to(device)
        return random_noise
    
    def EM_attack(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        for _ in range(self.num_steps):
           
            opt = torch.optim.Adam([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                if hasattr(model, 'classify'):
                    model.classify = True
                logits = model(perturb_img)
                loss = criterion(logits, labels)
            else:
                logits, loss = criterion(model, perturb_img, labels, optimizer)
            perturb_img.retain_grad()
            loss.backward()
            eta = self.step_size * perturb_img.grad.data.sign() * (-1)
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

    def datacook_v1(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        eta = random_noise
        first_outputs = model(perturb_img)
        _, first_predictions = torch.max(first_outputs, dim=1)
        if(len(first_predictions)==4):
            Pseudo = first_predictions.squeeze(0).long()
        else:
            Pseudo = first_predictions.long()
        
        for _ in range(self.num_steps):
            opt = torch.optim.Adam([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                if hasattr(model, 'classify'):
                    model.classify = True
                logits = model(perturb_img)
                loss = criterion(logits, Pseudo)
            else:
                logits, loss = criterion(model, perturb_img, Pseudo, optimizer)
            perturb_img.retain_grad()
            loss.backward()
            eta = self.step_size * perturb_img.grad.data.sign() * (-1)

            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

    def datacook_adv(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        first_outputs = model(perturb_img)
        _, first_predictions = torch.max(first_outputs, dim=1)
        if(len(first_predictions)==4):
            Pseudo = first_predictions.squeeze(0).long()
        else:
            Pseudo = first_predictions.long()
        
        for _ in range(self.num_steps):
            # opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt = torch.optim.Adam([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                if hasattr(model, 'classify'):
                    model.classify = True
                logits = model(perturb_img)
                loss = criterion(logits, Pseudo)
            else:
                logits, loss = criterion(model, perturb_img, Pseudo, optimizer)
            perturb_img.retain_grad()
            loss.backward()
            eta = self.step_size * perturb_img.grad.data.sign()
            # eta = self.step_size * perturb_img.grad.data.sign()
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            # eta = perturb_img.data - images.data
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

    def Adv_attack(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)
        
        labels = torch.squeeze(labels, 1).long().to(device)
        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        eta = random_noise
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                logits = model(perturb_img)
                loss = criterion(logits, labels)
            else:
                logits, loss = criterion(model, perturb_img, labels, optimizer)
            perturb_img.retain_grad()
            loss.backward()

            eta = self.step_size * perturb_img.grad.data.sign()
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

    def _patch_noise_extend_to_img(self, noise, image_size=[3, 32, 32], patch_location='center'):
        c, h, w = image_size[0], image_size[1], image_size[2]
        mask = np.zeros((c, h, w), np.float32)
        x_len, y_len = noise.shape[1], noise.shape[1]

        if patch_location == 'center' or (h == w == x_len == y_len):
            x = h // 2
            y = w // 2
        elif patch_location == 'random':
            x = np.random.randint(x_len // 2, w - x_len // 2)
            y = np.random.randint(y_len // 2, h - y_len // 2)
        else:
            raise('Invalid patch location')

        x1 = np.clip(x - x_len // 2, 0, h)
        x2 = np.clip(x + x_len // 2, 0, h)
        y1 = np.clip(y - y_len // 2, 0, w)
        y2 = np.clip(y + y_len // 2, 0, w)
        if type(noise) is np.ndarray:
            pass
        else:
            mask[:, x1: x2, y1: y2] = noise.cpu().numpy()
        return ((x1, x2, y1, y2), torch.from_numpy(mask).to(device))

    def _patch_noise_to_3d_img(self, noise, image_size=[3, 32, 32, 32], patch_location='center'):
        c, d, h, w = image_size[0], image_size[1], image_size[2], image_size[3]

        mask = np.zeros((c, d, h, w), np.float32)
        x_len, y_len, z_len = noise.shape[1], noise.shape[2], noise.shape[3]  # 假设 noise 是三维的

        if patch_location == 'center' or (d == h == w == x_len == y_len == z_len):
            x = d // 2
            y = h // 2
            z = w // 2
        elif patch_location == 'random':
            x = np.random.randint(x_len // 2, d - x_len // 2)
            y = np.random.randint(y_len // 2, h - y_len // 2)
            z = np.random.randint(z_len // 2, w - z_len // 2)
        else:
            raise ValueError('Invalid patch location')

        x1 = np.clip(x - x_len // 2, 0, d)
        x2 = np.clip(x + x_len // 2, 0, d)
        y1 = np.clip(y - y_len // 2, 0, h)
        y2 = np.clip(y + y_len // 2, 0, h)
        z1 = np.clip(z - z_len // 2, 0, w)
        z2 = np.clip(z + z_len // 2, 0, w)

        if isinstance(noise, np.ndarray):
            mask[:, x1: x2, y1: y2, z1: z2] = noise
        else:
            mask[:, x1: x2, y1: y2, z1: z2] = noise.cpu().numpy()

        return ((x1, x2, y1, y2, z1, z2), torch.from_numpy(mask).to(device))

    def comput_l2norm_lim(self, linf=0.03, feature_dim=3072):
        return np.sqrt(linf**2 * feature_dim)

    def normalize_l2norm(self, data, norm_lim):
        n = data.shape[0]
        orig_shape = data.shape
        flatten_data = data.reshape([n, -1])
        norms = np.linalg.norm(flatten_data, axis=1, keepdims=True)
        flatten_data = flatten_data/norms
        data = flatten_data.reshape(orig_shape)
        data = data * norm_lim
        return data

    def synthetic_perturbations(self, dataset, train_dataset, test_dataset):
        """
        Generate synthetic perturbations for training and testing datasets.

        The method first determines the number of samples required based on the dataset type.
        It then generates synthetic feature data using sklearn's `make_classification`, reshaping
        and repeating them to form image-like patches. The generated synthetic data is projected
        into a small L2-norm ball to control perturbation magnitude.

        Synthetic noise is then added to the original training and testing images, preserving
        class-wise alignment between synthetic data and original samples. Finally, the synthetic
        perturbations are converted to PyTorch tensors with proper shape for further use.

        Adapted from https://github.com/dayu11/Availability-Attacks-Create-Shortcuts
        """
        if dataset in ['PathMNIST', 'DermaMNIST', 'OCTMNIST', 'PneumoniaMNIST', 'RetinaMNIST', 'BreastMNIST', 'BloodMNIST', 'TissueMNIST', 'OrganAMNIST', 'OrganCMNIST', 'OrganSMNIST']:
            n = len(train_dataset)
            n *= 7
        else:
            n = train_dataset.data.shape[0] 

        img_size = self.imgsize
        noise_frame_size = self.patchsize
        info = INFO[dataset.lower()]
        num_classes = len(info['label'])

        min_val = np.min(train_dataset.imgs)
        max_val = np.max(train_dataset.imgs)

        is_even = img_size % noise_frame_size 
        num_patch = img_size//noise_frame_size 
        if(is_even > 0):
            num_patch += 1

        if(len(train_dataset.imgs.shape)==4):
            n_random_fea =  int((img_size/noise_frame_size)**2 * 3)

            # generate initial data points
            simple_data, simple_label = make_classification(n_samples=n, n_features=n_random_fea, n_classes=num_classes, n_informative=n_random_fea, n_redundant=0, n_repeated=0, class_sep=10., flip_y=0., n_clusters_per_class=1)
            simple_data = simple_data.reshape([simple_data.shape[0], num_patch, num_patch, 3])
            simple_data = simple_data.astype(np.float32)

            # duplicate each dimension to get 2-D patches
            simple_images = np.repeat(simple_data, noise_frame_size, 2) 
            simple_images = np.repeat(simple_images, noise_frame_size, 1)
            simple_data = simple_images[:, 0:img_size, 0:img_size, :]
        else:
            n_random_fea =  int((img_size/noise_frame_size)**2)

            # generate initial data points
            simple_data, simple_label = make_classification(n_samples=n, n_features=n_random_fea, n_classes=num_classes, n_informative=n_random_fea, n_redundant=0, n_repeated=0, class_sep=10., flip_y=0., n_clusters_per_class=1)
            simple_data = simple_data.reshape([simple_data.shape[0], num_patch, num_patch])
            simple_data = simple_data.astype(np.float32)

            # duplicate each dimension to get 2-D patches
            simple_images = np.repeat(simple_data, noise_frame_size, 2) 
            simple_images = np.repeat(simple_images, noise_frame_size, 1)
            simple_data = simple_images[:, 0:img_size, 0:img_size]

        # project the synthetic images into a small L2 ball
        linf = self.epsilon

        feature_dim = img_size**2 * 3
        l2norm_lim = self.comput_l2norm_lim(linf, feature_dim)
        simple_data = self.normalize_l2norm(simple_data, l2norm_lim)

        imgs_copy = train_dataset.imgs.copy()
        test_imgs_copy = test_dataset.imgs.copy()

        if imgs_copy.dtype != np.float32:
            imgs_copy = imgs_copy.astype(np.float32) / 255.0
            test_imgs_copy = test_imgs_copy.astype(np.float32) / 255.0

        arr_target = np.array(train_dataset.labels)
        arr_test_target = np.array(test_dataset.labels)

        perturb_img = imgs_copy.copy()
        perturb_test_img = test_imgs_copy.copy()

        # add synthetic noises to original examples
        for label in range(num_classes):
            orig_data_idx = arr_target == label
            simple_data_idx = simple_label == label
            
            mini_simple_data_for_train = simple_data[simple_data_idx][0:int(sum(orig_data_idx))]
            
            mini_simple_data_for_test = simple_data[simple_data_idx][int(sum(orig_data_idx)):int(sum(orig_data_idx)) + int(sum(arr_test_target == label))]
            
            orig_data_idx = np.squeeze(orig_data_idx, 1)
            perturb_img[orig_data_idx] += mini_simple_data_for_train
            
            test_data_idx = arr_test_target == label
            test_data_idx = np.squeeze(test_data_idx, 1)
            perturb_test_img[test_data_idx] += mini_simple_data_for_test

        random_noise = perturb_img - imgs_copy
        random_test_noise = perturb_test_img - test_imgs_copy

        if(len(imgs_copy.shape)==3):
            random_noise = np.stack((random_noise,)*3, axis=-1)
            random_test_noise = np.stack((random_test_noise,)*3, axis=-1)

        random_noise = torch.tensor(random_noise, dtype=torch.float32)
        random_test_noise = torch.tensor(random_test_noise, dtype=torch.float32)

        random_noise = random_noise.permute(0, 3, 1, 2)
        random_test_noise = random_test_noise.permute(0, 3, 1, 2)

        return random_noise, random_test_noise

    def synthetic_3dperturbations(self, dataset, train_dataset, test_dataset):
        n = len(train_dataset)
        n *= 7

        img_size = self.imgsize  
        depth_size = self.imgsize  
        noise_frame_size = self.patchsize  
        info = INFO[dataset.lower()]
        num_classes = len(info['label'])

        min_val = np.min(train_dataset.imgs)
        max_val = np.max(train_dataset.imgs)

        is_even = img_size % noise_frame_size 
        num_patch = img_size // noise_frame_size 
        num_depth_patch = depth_size // noise_frame_size
        if is_even > 0:
            num_patch += 1
            num_depth_patch += 1

        
        if len(train_dataset.imgs.shape) == 4: 
            n_random_fea = int((img_size / noise_frame_size) ** 2 * (depth_size / noise_frame_size))

  
            simple_data, simple_label = make_classification(
                n_samples=n, n_features=n_random_fea, n_classes=num_classes,
                n_informative=n_random_fea, n_redundant=0, n_repeated=0,
                class_sep=10., flip_y=0., n_clusters_per_class=1
            )
            simple_data = simple_data.reshape([simple_data.shape[0], num_depth_patch, num_patch, num_patch])
            simple_data = simple_data.astype(np.float32)

            
            simple_images = np.repeat(simple_data, noise_frame_size, axis=2)
            simple_images = np.repeat(simple_images, noise_frame_size, axis=3)
            simple_images = np.repeat(simple_images, noise_frame_size, axis=1)
            simple_data = simple_images[:, 0:depth_size, 0:img_size, 0:img_size]
        else:
            raise ValueError("输入数据不是 3D 图像格式")

        linf = self.epsilon
        feature_dim = depth_size * img_size ** 2
        l2norm_lim = self.comput_l2norm_lim(linf, feature_dim)
        simple_data = self.normalize_l2norm(simple_data, l2norm_lim)


        imgs_copy = train_dataset.imgs.copy()
        test_imgs_copy = test_dataset.imgs.copy()

        if imgs_copy.dtype != np.float32:
            imgs_copy = imgs_copy.astype(np.float32) / 255.0
            test_imgs_copy = test_imgs_copy.astype(np.float32) / 255.0

        arr_target = np.array(train_dataset.labels)
        arr_test_target = np.array(test_dataset.labels)

        perturb_img = imgs_copy.copy()
        perturb_test_img = test_imgs_copy.copy()

        for label in range(num_classes):
            orig_data_idx = arr_target == label
            simple_data_idx = simple_label == label

            mini_simple_data_for_train = simple_data[simple_data_idx][0:int(sum(orig_data_idx))]
、
            mini_simple_data_for_test = simple_data[simple_data_idx][int(sum(orig_data_idx)):int(sum(orig_data_idx)) + int(sum(arr_test_target == label))]

            orig_data_idx = np.squeeze(orig_data_idx, 1)
            perturb_img[orig_data_idx] += mini_simple_data_for_train

            test_data_idx = arr_test_target == label
            test_data_idx = np.squeeze(test_data_idx, 1)
            perturb_test_img[test_data_idx] += mini_simple_data_for_test

        random_noise = perturb_img - imgs_copy
        random_test_noise = perturb_test_img - test_imgs_copy

        random_noise = torch.tensor(random_noise, dtype=torch.float32)
        random_test_noise = torch.tensor(random_test_noise, dtype=torch.float32)

        random_noise = random_noise.permute(0, 1, 2, 3)
        random_test_noise = random_test_noise.permute(0, 1, 2, 3)

        return random_noise, random_test_noise