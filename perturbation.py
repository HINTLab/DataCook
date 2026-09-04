import argparse
import datetime
import os
import shutil
import time
import torch
import numpy as np
from pathlib import Path
from medmnist import INFO
from tqdm import tqdm

import mlconfig
import util
import dataset
import toolbox

from trainer import Trainer
from evaluator import Evaluator


from sklearn.cluster import KMeans
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim

def parse_args():
    """
    Parse command-line arguments for experimental configuration.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Fingerprint Framework')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--version', type=str, default="resnet18", help='Model version name')
    parser.add_argument('--exp_name', type=str, default="test_exp", help='Experiment name')
    parser.add_argument('--config_path', type=str, default='configs/MedMNIST', help='Path to configuration YAML')
    parser.add_argument('--load_model', action='store_true', help='Whether to load pretrained model')
    parser.add_argument('--data_parallel', action='store_true', help='Use DataParallel for multi-GPU')

    # Dataset options
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--num_of_workers', type=int, default=8)
    parser.add_argument('--train_data_type', type=str, default='PathMNIST')
    parser.add_argument('--train_data_path', type=str, default='../datasets')
    parser.add_argument('--test_data_type', type=str, default='PathMNIST')
    parser.add_argument('--test_data_path', type=str, default='../datasets')
    parser.add_argument('--recover_test_data_type', type=str, default='PathMNIST')
    parser.add_argument('--recover_test_data_path', type=str, default='../datasets')

    # fingerprint options
    parser.add_argument('--universal_stop_error', type=float, default=0.5)
    parser.add_argument('--train_step', default=10, type=int, help='Number of training steps for each epoch, only for EM-Pseudo and EM')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--attack_type', type=str, default='datacook', choices=['EM', 'Adv', 'random', 'Unlearnable_Cluster', 'lsp', 'datacook', 'datacook_adv', 'EM-Pseudo'], help='Attack type')
    parser.add_argument('--patch_location', type=str, default='center', choices=['center', 'random'])
    parser.add_argument('--noise_shape', type=int, nargs='+', default=[10, 3, 32, 32])
    parser.add_argument('--test_noise_shape', type=int, nargs='+', default=[10, 3, 32, 32])
    parser.add_argument('--epsilon', type=float, default=8, help='Max size of fingerprint')
    parser.add_argument('--num_steps', type=int, default=1, help='Number of PGD steps')
    parser.add_argument('--step_size', type=float, default=0.8, help='Step size for PGD')
    
    # clusters
    parser.add_argument('--num_clusters', default=10, type=int)
    parser.add_argument('--g_net_lr', default=1e-2, type=float)
    parser.add_argument('--g_net_num_epoch', default=50, type=int)
    return parser.parse_args()

def setup_environment(args):
    """
    Setup experiment directories, logger, and device.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        tuple: exp_path, checkpoint_path, fingerprint_path, logger, device, num_classes
    """
    # Convert epsilon and step size from pixel range to [0,1]
    args.epsilon /= 255.0
    args.step_size /= 255.0

    # Setup experiment directories  
    if args.exp_name == '':
        args.exp_name = 'exp_' + datetime.datetime.now()
    exp_path = os.path.join(args.exp_name, args.version)
    os.makedirs(exp_path, exist_ok=True)
    
    checkpoint_path = os.path.join(exp_path, 'checkpoints')
    fingerprint_path = os.path.join(exp_path, 'fingerprint')
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(fingerprint_path, exist_ok=True)

    # Setup logger
    logger = util.setup_logger(name=args.version, log_file=os.path.join(exp_path, args.version + ".log"))

    # Setup device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda')
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    # Log basic info
    logger.info("Experiment Path: %s", exp_path)
    logger.info("PyTorch Version: %s", torch.__version__)

    # Get number of classes from MedMNIST info
    info = INFO[args.train_data_type.lower()]
    num_classes = len(info['label'])

    return exp_path, checkpoint_path, fingerprint_path, logger, device, num_classes

def train(args, config, logger, model, optimizer, scheduler, criterion, trainer, evaluator, ENV, ckpt_path):
    """
    Pretrain the model to extract robust data features.
    If the pretrain checkpoint exists, load it and evaluate once.
    Otherwise, run the pretraining loop and save after each epoch.
    """
    model_file = os.path.join(ckpt_path, f"{args.version}_{config.epochs}.pth")
    logger.info('Model Saved at %s', model_file)

    if os.path.exists(model_file):
        latest_model_path = model_file
        logger.info("Loading model from %s", latest_model_path)
        checkpoint = torch.load(latest_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("Evaluating model without fine-tuning...")
        evaluator.eval_model(0, model)
        payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
        logger.info(payload)
        return 

    else:
        logger.info("Starting or continuing training process...")
        for epoch in range(config.epochs):
            logger.info("")
            logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)

            # Train
            ENV['global_step'] = trainer.train_model(epoch, model, criterion, optimizer)
            ENV['train_history'].append(trainer.acc_meters.avg*100)
            scheduler.step()

            # Eval
            logger.info("="*20 + "Eval Epoch %d" % (epoch) + "="*20)
            evaluator.eval_model(epoch, model)
            payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
            logger.info(payload)
            ENV['eval_history'].append(evaluator.acc_meters.avg*100)
            ENV['curren_acc'] = evaluator.acc_meters.avg*100

            # Reset Stats
            trainer._reset_stats()
            evaluator._reset_stats()

            # Save Model
            target_model = model.module if args.data_parallel else model
            util.save_model(ENV=ENV,
                            epoch=epoch,
                            model=target_model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            filename=model_file)
            logger.info('Model Saved at %s', model_file)
        return

def check_ssim(lsp_noise, lsp_test_noise, data_loader, num_classes, perturbation_path, device, logger):
    """
    Compute SSIM for LSP-generated perturbations on train and test sets,
    save perturbed images & noise maps, and log average SSIM.
    """
    def _process_dataset(dataset, noise, prefix):
        idx = 0
        count = [0 for _ in range(num_classes)]

        # Create output dirs
        for i in range(num_classes):
            Path(os.path.join(perturbation_path, '..', f'{prefix}', str(i))).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(perturbation_path, '..', f'{prefix}_noisy', str(i))).mkdir(parents=True, exist_ok=True)

        ssim_total = 0
        num_images = 0

        for images, labels in dataset:
            images, labels = images.to(device), labels.to(device)
            for image, label in zip(images, labels):
                sample_noise = torch.tensor(noise[idx]).to(device) if not torch.is_tensor(noise) else noise[idx].to(device)
                perturb_img = torch.clamp(image + sample_noise, 0, 1)
                gt = label.item()

                save_image(perturb_img, os.path.join(perturbation_path, '..', f'{prefix}', str(gt), f'{count[gt]}.png'))
                save_image(sample_noise, os.path.join(perturbation_path, '..', f'{prefix}_noisy', str(gt), f'{count[gt]}.png'))
                count[gt] += 1
                idx += 1

                # Compute SSIM
                img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                perturb_img_np = perturb_img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                current_ssim = ssim(img_np, perturb_img_np, channel_axis=2, data_range=1)
                ssim_total += current_ssim
                num_images += 1

        avg_ssim = ssim_total / num_images
        logger.info(f'Average {prefix} SSIM: {avg_ssim:.4f}')
        return noise
    # Process train and test datasets
    random_noise = _process_dataset(data_loader['train_dataset'], lsp_noise, 'ae')
    random_test_noise = _process_dataset(data_loader['test_dataset'], lsp_test_noise, 'ae_test')

    return random_noise, random_test_noise

def generate(noise_generator, perturbation_path, cluster_path_file, epsilon, data_loader, random_noise, random_test_noise, ENV, num_classes, logger):
    """
    Generate UC perturbations, apply to datasets, compute SSIM, save results.
    """    
    cluster = torch.load(cluster_path_file, map_location='cpu')
    num_clusters = cluster['centers'].shape[0]
    pred_idx_map = {
        'train_dataset': cluster['pred_idx'],
        'test_dataset': cluster['pred_idx_test']
    }

    # Load and clamp noise
    noise = torch.cat([
        torch.load(os.path.join(perturbation_path, f'perturbation_{i}.pth'), map_location='cpu')['perturbation']
        for i in range(num_clusters)
    ], dim=0)
    noise = torch.clamp(noise, 0, epsilon)

    def _process_dataset(name, pred_idx, noise_store):
        idx = 0
        count = [0] * num_classes
        Path(os.path.join(perturbation_path, '..', f'{name}')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(perturbation_path, '..', f'{name}_noisy')).mkdir(parents=True, exist_ok=True)

        ssim_total, num_images = 0, 0
        for images, labels in data_loader[name]:
            for image, label in zip(images, labels):
                img_size = image.size()
                eta = torch.nn.functional.interpolate(noise[pred_idx[idx]].unsqueeze(0), (img_size[1], img_size[2]))
                perturb_img = torch.clamp(image + eta, 0, 1)

                noise_store[idx] = perturb_img - image
                gt = label.tolist()[0]

                save_image(perturb_img, os.path.join(perturbation_path, '..', f'{name}', str(gt), f'{count[gt]}.png'))
                save_image(noise_store[idx], os.path.join(perturbation_path, '..', f'{name}_noisy', str(gt), f'{count[gt]}.png'))
                count[gt] += 1
                idx += 1

                img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                perturb_img_np = perturb_img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                ssim_total += ssim(img_np, perturb_img_np, channel_axis=2, data_range=1)
                num_images += 1

        logger.info(f'Average {name} SSIM: {ssim_total / num_images:.4f}')

    # Process train/test
    _process_dataset('train_dataset', pred_idx_map['train_dataset'], random_noise)
    _process_dataset('test_dataset', pred_idx_map['test_dataset'], random_test_noise)

    # Save to ENV
    if torch.is_tensor(random_noise):
        ENV['random_noise'] = random_noise
        ENV['random_test_noise'] = random_test_noise

    return random_noise, random_test_noise

def train_gnet(model, g_net_lr, g_net_num_epoch, epsilon, data_loader, trainer, cluster_path_file, ENV, logger, perturbation_path, target_offset=1):
    """
    Train the generator network (G-Net) used in the Unlearnable Clusters (UC) method
    for feature-specific perturbation generation.

    This function is adapted from:
        https://github.com/jiamingzhang94/Unlearnable-Clusters
    """   
    cluster = torch.load(cluster_path_file, map_location='cpu')
    num_clusters = cluster['centers'].shape[0]

    trainer.gnet_train(num_clusters, cluster, g_net_num_epoch, g_net_lr, model, epsilon, logger, perturbation_path, data_loader, target_offset)

def data_cluster(num_clusters, dataset_name, model_name, model, data_loader, device, logger, cluster_path):
    """
    Build feature-space clusters for the Unlearnable Clusters (UC) method.

    Adapted from:
        https://github.com/jiamingzhang94/Unlearnable-Clusters

    This function extracts intermediate feature representations from a 
    pre-trained model, applies KMeans clustering to group samples into 
    `num_clusters`, and stores both training and test sample assignments 
    along with cluster centers. These cluster assignments are later used 
    to generate cluster-specific perturbations in the UC pipeline.
    """
    logger.info("")
    logger.info("="*20 + "Build Clusters " + "="*20)

    # Train
    features = []
    def hook(layer, inp, out):
        features.append(inp[0].cpu())
    hook_handle = model.fc.register_forward_hook(hook)

    for i, (images, labels) in enumerate(data_loader["train_dataset"]):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.no_grad():
            model(images)
    features = torch.cat(features, dim=0)
    classifier = KMeans(n_clusters=num_clusters)
    pred_idx = classifier.fit_predict(features.numpy())

    features = []
    for i, (images, labels) in enumerate(data_loader["test_dataset"]):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.no_grad():
            model(images)
    features = torch.cat(features, dim=0)
    pred_idx_test = classifier.predict(features.numpy())
    hook_handle.remove()
    logger.info('pred_idx')
    logger.info(pred_idx)
    logger.info('pred_idx_test')
    logger.info(pred_idx_test)

    # Save Model
    result = {'pred_idx': torch.tensor(pred_idx), 'pred_idx_test': torch.tensor(pred_idx_test), 'centers': torch.tensor(classifier.cluster_centers_)}
    num_clusters = result['centers'].shape[0]
    logger.info(classifier.cluster_centers_)
    logger.info('num_clusters %d', num_clusters)
    cluster_path_file = os.path.join(cluster_path, f"{dataset_name}_{model_name.lower()}_cluster{num_clusters}.pth")
    torch.save(result, cluster_path_file)
    logger.info('cluster Saved at %s', cluster_path_file)

    return cluster_path_file

def sample_wise_perturbation(args, logger, num_classes, perturbation_path,  noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise,random_test_noise, device, ENV):
    """
    Main routine for generating sample-wise perturbations.

    This function iteratively searches for optimal perturbations on a per-sample basis.
    It works by:
        1. Initializing patch coordinates for each training and test image.
        2. Iteratively updating perturbations (fingerprints) for each image, based on the selected attack type:
            - 'datacook'      : DataCook anti-adversarial fingerprints.
            - 'datacook_adv'  : Adversarial variant of DataCook.
            - 'Adv'           : Standard adversarial attack.
        3. Evaluating stopping conditions based on error rate or a maximum number of iterations.
        4. Saving final perturbations to disk for deployment.

    This method forms the core step for embedding sample-specific perturbations into the dataset.
    """
    # ---------------------------
    # 1. Prepare dataset loaders
    # ---------------------------
    datasets_generator = dataset.DatasetGenerator(train_batch_size=args.train_batch_size,
                                                  eval_batch_size=args.eval_batch_size,
                                                  train_data_type=args.train_data_type,
                                                  train_data_path=args.train_data_path,
                                                  test_data_type=args.test_data_type,
                                                  test_data_path=args.test_data_path,
                                                  recover_test_data_path=args.recover_test_data_path, 
                                                  recover_test_data_type=args.recover_test_data_type,
                                                  num_of_workers=args.num_of_workers,
                                                  seed=args.seed, no_train_augments=True)


    data_loader = datasets_generator.getDataLoader(train_shuffle=False, train_drop_last=False)
    mask_cord_list, test_mask_cord_list = [], []
    train_loader = data_loader['train_dataset']
    test_loader = data_loader['test_dataset']

    # ---------------------------
    # 2. Generate patch coordinates for each image
    # ---------------------------
    for ds, noise, cord_list in [(train_loader, random_noise, mask_cord_list),
                                (test_loader, random_test_noise, test_mask_cord_list)]:
        idx = 0
        for images, _ in ds:
            for image in images:
                patch_cord, _ = noise_generator._patch_noise_extend_to_img(noise[idx], image.shape, args.patch_location)
                cord_list.append(patch_cord)
                idx += 1

    # ---------------------------
    # 3. Iterative perturbation search loop
    # ---------------------------
    condition = True
    step_count = 0
    while condition:
        logger.info('=' * 20 + 'Searching Samplewise Perturbation' + '=' * 20)
        logger.info(f'Perturbation Search Iteration {step_count + 1}')
    
        for _, loader, fingerprint, cords in tqdm([('train', train_loader, random_noise, mask_cord_list),
                                        ('test', test_loader, random_test_noise, test_mask_cord_list)]):
            idx = 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                batch_fingerprint, start_idx = [], idx

                for i, image in enumerate(images):
                    c, h, w = image.shape
                    mask = torch.zeros((c, h, w), dtype=torch.float32)
                    x1, x2, y1, y2 = cords[idx]
                    patch = fingerprint[idx]
                    if isinstance(patch, np.ndarray):
                        patch = torch.tensor(patch)
                    mask[:, x1:x2, y1:y2] = patch
                    batch_fingerprint.append(mask)
                    idx += 1

                batch_fingerprint = torch.stack(batch_fingerprint).to(device)
                model.eval()
     
                for param in model.parameters():
                    param.requires_grad = False
                if args.attack_type == 'datacook':
                    _, delta = noise_generator.datacook_v1(images, labels, model, optimizer, criterion, random_noise= batch_fingerprint)
                elif args.attack_type == 'datacook_adv':
                    _, delta = noise_generator.datacook_adv(images, labels, model, optimizer, criterion, random_noise = batch_fingerprint)
                elif args.attack_type == 'Adv':
                    _, delta = noise_generator.Adv_attack(images, labels, model, optimizer, criterion, random_noise=batch_fingerprint)  
                else:
                    raise ValueError(f"Unknown attack type {args.attack_type}")

                for i, d in enumerate(delta):
                    x1, x2, y1, y2 = cords[start_idx + i]
                    d_patch = d[:, x1:x2, y1:y2]
                    fingerprint[start_idx + i] = d_patch.detach().cpu()

        # ---------------------------
        # 4. Evaluate stopping conditions
        # ---------------------------
        loss_avg, error_rate = samplewise_perturbation_eval(args, random_noise, random_test_noise, data_loader, model, device, eval_target='train_dataset',
                                                            mask_cord_list=mask_cord_list)
        logger.info('Loss: {:.4f} Acc: {:.2f}%'.format(loss_avg, 100 - error_rate*100))
        step_count+=1
        if torch.is_tensor(random_noise):
            random_noise = random_noise.detach()
            random_test_noise = random_test_noise.detach()
            ENV['random_noise'] = random_noise
            ENV['random_test_noise'] = random_test_noise
        if args.attack_type == 'datacook':
            condition = (step_count <= 5) and (error_rate > args.universal_stop_error)
        elif args.attack_type == 'Adv':
            condition =  (step_count <= 5) and (error_rate < args.universal_stop_error)
        elif args.attack_type == 'datacook_adv':
            condition = (step_count <= 5) and (error_rate < args.universal_stop_error)

    # ---------------------------
    # 5. Save final sample-wise perturbations
    # ---------------------------
    add_samplewise_perturbation(args, logger, num_classes, perturbation_path, random_noise, random_test_noise, data_loader, device, mask_cord_list=mask_cord_list, test_mask_cord_list= test_mask_cord_list) 

    return random_noise, random_test_noise

def add_samplewise_perturbation(args, logger, num_classes, perturbation_path, random_noise, random_test_noise, data_loader, device, mask_cord_list=[], test_mask_cord_list=[]):
    """
    Apply finalized sample-wise perturbations to datasets (train/test),
    save perturbed images & noise maps, and compute average SSIM.

    """

    # Datasets that need label squeeze from (B,1) -> (B,)
    MED2D = {
        'PathMNIST','DermaMNIST','OCTMNIST','PneumoniaMNIST','RetinaMNIST',
        'BreastMNIST','BloodMNIST','TissueMNIST','OrganAMNIST','OrganCMNIST','OrganSMNIST'
    }

    def _ensure_labels_shape(name, labels):
        if args.train_data_type in MED2D:
            return torch.squeeze(labels, 1).long()
        return labels

    def _mkdir_split(prefix):
        for cls in range(num_classes):
            Path(os.path.join(perturbation_path, '..', f'{prefix}', str(cls))).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(perturbation_path, '..', f'{prefix}_noisy', str(cls))).mkdir(parents=True, exist_ok=True)

    def _process_split(loader_key, noise_bank, coords, prefix):
        """Apply noise patches for a given split, save outputs, return avg SSIM."""
        if noise_bank is None:
            logger.info(f"Skip {prefix}: noise_bank is None")
            return 0.0

        _mkdir_split(prefix)
        idx = 0
        ssim_total, num_images = 0.0, 0

        for images, labels in data_loader[loader_key]:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            labels = _ensure_labels_shape(loader_key, labels).to(device)

            for j, (image, label) in enumerate(zip(images, labels)):
                # prepare patch-only noise mask
                c, h, w = image.shape
                sample_noise = torch.tensor(noise_bank[idx]).to(device) if not torch.is_tensor(noise_bank) else noise_bank[idx].to(device)
                x1, x2, y1, y2 = coords[idx]
                mask = np.zeros((c, h, w), np.float32)
                mask[:, x1:x2, y1:y2] = sample_noise.detach().cpu().numpy()
                sample_noise = torch.from_numpy(mask).to(device)

                # apply noise
                images[j] = images[j] + sample_noise

                # save
                gt = label.item()
                save_image(images[j], os.path.join(perturbation_path, '..', f'{prefix}', str(gt), f'{j}_{idx}.png'))
                save_image(sample_noise, os.path.join(perturbation_path, '..', f'{prefix}_noisy', str(gt), f'{j}_{idx}.png'))

                # compute SSIM (original image ≈ images[j] - sample_noise)
                img_np = (images[j] - sample_noise).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                pert_np = images[j].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                ssim_total += ssim(img_np, pert_np, channel_axis=2, data_range=1)
                num_images += 1
                idx += 1

        avg = ssim_total / max(1, num_images)
        logger.info(f'Average {prefix} SSIM: {avg:.4f}')
        return avg

    # Train split
    train_ssim = _process_split('train_dataset', random_noise, mask_cord_list, 'ae')

    # Test split
    test_ssim  = _process_split('test_dataset', random_test_noise, test_mask_cord_list, 'ae_test')

    return
    
def sample_wise_perturbation_min_min(args, logger, num_classes, perturbation_path, noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise,random_test_noise, device, ENV):
    """
    Adapted from https://github.com/HanxunH/Unlearnable-Examples
    Implements sample-wise perturbation search for EM and EM-Pseudo attacks.
    The goal is to iteratively update noise patterns for each training and test sample
    until the target error condition is reached.
    """

    datasets_generator = dataset.DatasetGenerator(train_batch_size=args.train_batch_size,
                                                  eval_batch_size=args.eval_batch_size,
                                                  train_data_type=args.train_data_type,
                                                  train_data_path=args.train_data_path,
                                                  test_data_type=args.test_data_type,
                                                  test_data_path=args.test_data_path,
                                                  recover_test_data_path=args.recover_test_data_path, 
                                                  recover_test_data_type=args.recover_test_data_type,
                                                  num_of_workers=args.num_of_workers,
                                                  seed=args.seed, no_train_augments=True)


    data_loader = datasets_generator.getDataLoader(train_shuffle=False, train_drop_last=False)
    mask_cord_list = []
    test_mask_cord_list = []

    idx = 0
    for images, labels in data_loader['train_dataset']:
        for i, (image, label) in enumerate(zip(images, labels)):
            noise = random_noise[idx]
            mask_cord, _ = noise_generator._patch_noise_extend_to_img(noise, image_size=image.shape, patch_location=args.patch_location)
            mask_cord_list.append(mask_cord)
            idx += 1

    idx = 0
    for images, labels in data_loader['test_dataset']:
        for i, (image, label) in enumerate(zip(images, labels)):
            test_noise = random_test_noise[idx]
            mask_cord, _ = noise_generator._patch_noise_extend_to_img(test_noise, image_size=image.shape, patch_location=args.patch_location)
            test_mask_cord_list.append(mask_cord)
            idx += 1

    condition = True
    train_idx = 0

    data_iter = iter(data_loader['train_dataset'])
    number_attack = 0
    while condition:
        if (args.attack_type == 'EM' or args.attack_type == 'EM-Pseudo') and not args.load_model:
            for j in tqdm(range(0, args.train_step), total=args.train_step):
                try:
                    (images, labels) = next(data_iter)
                except:
                    train_idx = 0
                    data_iter = iter(data_loader['train_dataset'])
                    (images, labels) = next(data_iter)

                images, labels = images.to(device), labels.to(device)

                if args.train_data_type == 'PathMNIST' or args.train_data_type == 'DermaMNIST' or args.train_data_type == 'OCTMNIST' or args.train_data_type == 'PneumoniaMNIST' or args.train_data_type == 'RetinaMNIST' or  args.train_data_type == 'BreastMNIST' or args.train_data_type == 'BloodMNIST' or args.train_data_type == 'TissueMNIST' or args.train_data_type == 'OrganAMNIST' or args.train_data_type == 'OrganCMNIST' or args.train_data_type == 'OrganSMNIST':
                    labels = torch.squeeze(labels, 1).long().to(device)

                # Add Sample-wise Noise to each sample
                for i, (image, label) in enumerate(zip(images, labels)):
                    sample_noise = random_noise[train_idx]
                    c, h, w = image.shape[0], image.shape[1], image.shape[2]
                    mask = np.zeros((c, h, w), np.float32)
                    x1, x2, y1, y2 = mask_cord_list[train_idx]
                    if type(sample_noise) is np.ndarray:
                        mask[:, x1: x2, y1: y2] = sample_noise
                    else:
                        mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                    # mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                    sample_noise = torch.from_numpy(mask).to(device)
                    images[i] = images[i] + sample_noise
                    train_idx += 1

                model.train()
                for param in model.parameters():
                    param.requires_grad = True
                trainer.train_batch(images, labels, model, optimizer)

        # Search For Noise
        idx = 0
        for i, (images, labels) in tqdm(enumerate(data_loader['train_dataset']), total=len(data_loader['train_dataset'])):
            images, labels, model = images.to(device), labels.to(device), model.to(device)

            if args.train_data_type == 'PathMNIST' or args.train_data_type == 'DermaMNIST' or args.train_data_type == 'OCTMNIST' or args.train_data_type == 'PneumoniaMNIST' or args.train_data_type == 'RetinaMNIST' or  args.train_data_type == 'BreastMNIST' or args.train_data_type == 'BloodMNIST' or args.train_data_type == 'TissueMNIST' or args.train_data_type == 'OrganAMNIST' or args.train_data_type == 'OrganCMNIST' or args.train_data_type == 'OrganSMNIST':
                labels = torch.squeeze(labels, 1).long().to(device)
            # Add Sample-wise Noise to each sample
            batch_noise, batch_start_idx = [], idx
            for i, (image, label) in enumerate(zip(images, labels)):
                sample_noise = random_noise[idx]
                
                c, h, w = image.shape[0], image.shape[1], image.shape[2]
                mask = np.zeros((c, h, w), np.float32)
                
                x1, x2, y1, y2 = mask_cord_list[idx]
                if type(sample_noise) is np.ndarray:
                    mask[:, x1: x2, y1: y2] = sample_noise
                else:
                    mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()

                sample_noise = torch.from_numpy(mask).to(device)
            
                batch_noise.append(sample_noise)
                idx += 1

            # Update sample-wise perturbation
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            batch_noise = torch.stack(batch_noise).to(device)
            if args.attack_type == 'EM':
                perturb_img, eta = noise_generator.EM_attack(images, labels, model, optimizer, criterion, random_noise=batch_noise)
            elif args.attack_type == 'EM-Pseudo':
                perturb_img, eta = noise_generator.datacook_v1(images, labels, model, optimizer, criterion, random_noise=batch_noise)
            else:
                raise('Invalid attack')

            for i, delta in enumerate(eta):
                x1, x2, y1, y2 = mask_cord_list[batch_start_idx+i]
                delta = delta[:, x1: x2, y1: y2]
                if torch.is_tensor(random_noise):
                    random_noise[batch_start_idx+i] = delta.detach().cpu().clone()
                else:
                    random_noise[batch_start_idx+i] = delta.detach().cpu().numpy()

       # Search For test Noise
        idx = 0
        for i, (images, labels) in tqdm(enumerate(data_loader['test_dataset']), total=len(data_loader['test_dataset'])):
            images, labels, model = images.to(device), labels.to(device), model.to(device)

            if args.train_data_type == 'PathMNIST' or args.train_data_type == 'DermaMNIST' or args.train_data_type == 'OCTMNIST' or args.train_data_type == 'PneumoniaMNIST' or args.train_data_type == 'RetinaMNIST' or  args.train_data_type == 'BreastMNIST' or args.train_data_type == 'BloodMNIST' or args.train_data_type == 'TissueMNIST' or args.train_data_type == 'OrganAMNIST' or args.train_data_type == 'OrganCMNIST' or args.train_data_type == 'OrganSMNIST':
                labels = torch.squeeze(labels, 1).long().to(device)

            # Add Sample-wise Noise to each sample
            batch_test_noise, batch_test_start_idx = [], idx
            for i, (image, label) in enumerate(zip(images, labels)):
                sample_noise = random_test_noise[idx]
                c, h, w = image.shape[0], image.shape[1], image.shape[2]
                mask = np.zeros((c, h, w), np.float32)
                x1, x2, y1, y2 = mask_cord_list[idx]
                if type(sample_noise) is np.ndarray:
                    mask[:, x1: x2, y1: y2] = sample_noise
                else:
                    mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                sample_noise = torch.from_numpy(mask).to(device)
                batch_test_noise.append(sample_noise)
                idx += 1

            # Update sample-wise perturbation
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            batch_test_noise = torch.stack(batch_test_noise).to(device)
            if args.attack_type == 'EM':
                perturb_img, eta = noise_generator.EM_attack(images, labels, model, optimizer, criterion, random_noise=batch_test_noise)
            elif args.attack_type == 'EM-Pseudo':
                perturb_img, eta = noise_generator.datacook_v1(images, labels, model, optimizer, criterion, random_noise=batch_test_noise)            
            else:
                raise('Invalid attack')

            for i, delta in enumerate(eta):
                x1, x2, y1, y2 = test_mask_cord_list[batch_test_start_idx+i]
                delta = delta[:, x1: x2, y1: y2]
                if torch.is_tensor(random_test_noise):
                    random_test_noise[batch_test_start_idx+i] = delta.detach().cpu().clone()
                else:
                    random_test_noise[batch_test_start_idx+i] = delta.detach().cpu().numpy()

        # Eval termination conditions
        loss_avg, error_rate = samplewise_perturbation_eval(args, random_noise, random_test_noise, data_loader, model, device, eval_target='train_dataset',
                                                            mask_cord_list=mask_cord_list)
        logger.info('Loss: {:.4f} Acc: {:.2f}%'.format(loss_avg, 100 - error_rate*100))
        number_attack+=1
        if torch.is_tensor(random_noise):
            random_noise = random_noise.detach()
            random_test_noise = random_test_noise.detach()
            ENV['random_noise'] = random_noise
            ENV['random_test_noise'] = random_test_noise
        if args.attack_type == 'EM':
            condition = error_rate > args.universal_stop_error
        elif args.attack_type == 'EM-Pseudo':
            condition = (number_attack <= 5) and (error_rate > args.universal_stop_error)

    add_samplewise_perturbation(args, logger, num_classes, perturbation_path, random_noise, random_test_noise, data_loader, device, eval_target='train_dataset', mask_cord_list=mask_cord_list) 

    return random_noise, random_test_noise

def samplewise_perturbation_eval(args, random_noise, random_test_noise, data_loader, model, device, eval_target='train_dataset', mask_cord_list=[]):
    loss_meter = util.AverageMeter()
    err_meter = util.AverageMeter()

    model = model.to(device)
    idx = 0

    # Datasets that need label squeeze from (B,1) -> (B,)
    MED2D = {
        'PathMNIST','DermaMNIST','OCTMNIST','PneumoniaMNIST','RetinaMNIST',
        'BreastMNIST','BloodMNIST','TissueMNIST','OrganAMNIST','OrganCMNIST','OrganSMNIST'
    }

    def _ensure_labels_shape(name, labels):
        if args.train_data_type in MED2D:
            return torch.squeeze(labels, 1).long()
        return labels

    for i, (images, labels) in enumerate(data_loader[eval_target]):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        labels = _ensure_labels_shape(loader_key, labels).to(device)    
            
        if random_noise is not None:
            for j, (image, label) in enumerate(zip(images, labels)):
                if not torch.is_tensor(random_noise):
                    sample_noise = torch.tensor(random_noise[idx]).to(device)
                else:
                    sample_noise = random_noise[idx].to(device)
                c, h, w = image.shape[0], image.shape[1], image.shape[2]
                mask = np.zeros((c, h, w), np.float32)
                x1, x2, y1, y2 = mask_cord_list[idx]
                mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                sample_noise = torch.from_numpy(mask).to(device)
                images[j] = images[j] + sample_noise
                idx += 1
        
        pred = model(images)
        err = (pred.data.max(1)[1] != labels.data).float().sum()
        loss = torch.nn.CrossEntropyLoss()(pred, labels)
        loss_meter.update(loss.item(), len(labels))
        err_meter.update(err / len(labels))
       
    return loss_meter.avg, err_meter.avg

def main():
    """
    Main entry point for fingerprint generation.
    Initializes dataset, model, optimizer, and starts the fingerprint process.
    """
    args = parse_args()
    exp_path, ckpt_path, fingerprint_path, logger, device, num_classes = setup_environment(args)

    # Load configuration
    config_file = os.path.join(args.config_path, args.version + '.yaml')
    config = mlconfig.load(config_file)
    config.set_immutable()
    shutil.copyfile(config_file, os.path.join(exp_path, args.version + '.yaml'))

    datasets_generator = dataset.DatasetGenerator(train_batch_size=args.train_batch_size,
                                                  eval_batch_size=args.eval_batch_size,
                                                  train_data_type=args.train_data_type,
                                                  train_data_path=args.train_data_path,
                                                  test_data_type=args.test_data_type,
                                                  test_data_path=args.test_data_path,
                                                  recover_test_data_path=args.recover_test_data_path, 
                                                  recover_test_data_type=args.recover_test_data_type,
                                                  num_of_workers=args.num_of_workers,
                                                  seed=args.seed)


    data_loader = datasets_generator.getDataLoader(train_shuffle=False, train_drop_last=False)
    model = config.model().to(device)
 
    logger.info("param size = %fMB", util.count_parameters_in_MB(model))
    optimizer = config.optimizer(model.parameters())
    scheduler = config.scheduler(optimizer)
    criterion = config.criterion()

    trainer = Trainer(criterion, data_loader, logger, config, args.train_data_type, target='train_dataset')
    evaluator = Evaluator(data_loader, logger, config, args.train_data_type)

    ENV = {'global_step': 0,
           'best_acc': 0.0,
           'curren_acc': 0.0,
           'best_pgd_acc': 0.0,
           'train_history': [],
           'eval_history': [],
           'pgd_eval_history': [],
           'genotype_list': []}

    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    if args.load_model:
        checkpoint = util.load_model(filename=checkpoint_path_file,
                                     model=model,
                                     optimizer=optimizer,
                                     alpha_optimizer=None,
                                     scheduler=scheduler)
        ENV = checkpoint['ENV']
        trainer.global_step = ENV['global_step']
        logger.info("File %s loaded!" % (checkpoint_path_file))

    noise_generator = toolbox.PerturbationTool(epsilon=args.epsilon,
                                               num_steps=args.num_steps,
                                               step_size=args.step_size)

    random_noise = torch.zeros(*args.noise_shape)
    random_test_noise = torch.zeros(*args.test_noise_shape)

    if args.attack_type == 'random':
        noise = noise_generator.random_noise(noise_shape=args.noise_shape)
        test_noise = noise_generator.random_noise(noise_shape=args.test_noise_shape)     

    elif args.attack_type == 'Unlearnable_Cluster':
        train(args, config, logger, model, optimizer, scheduler, criterion, trainer, evaluator, ENV, ckpt_path)
        cluster_path = os.path.join(exp_path, 'cluster')
        os.makedirs(cluster_path, exist_ok=True)
        cluster_path_file = data_cluster(args.num_clusters, args.train_data_type, args.version, model, data_loader, device, logger, cluster_path)
        train_gnet(model, args.g_net_lr, args.g_net_num_epoch, args.epsilon, data_loader, trainer, cluster_path_file, ENV, logger, fingerprint_path)

        noise, test_noise = generate(noise_generator, fingerprint_path, cluster_path_file, args.epsilon, data_loader, random_noise, random_test_noise, ENV, num_classes, logger)

    elif args.attack_type == 'lsp':
        noise, test_noise =  noise_generator.synthetic_perturbations(args.train_data_type, datasets_generator.datasets['train_dataset'], datasets_generator.datasets['test_dataset'])
        check_ssim(noise, test_noise, data_loader, num_classes, fingerprint_path, device, logger)

    elif args.attack_type in ['datacook', 'datacook_adv', 'Adv']:

        train(args, config, logger, model, optimizer, scheduler, criterion, trainer, evaluator, ENV, ckpt_path)
        evaluator.eval_model(0, model)
        payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
        logger.info(payload)
        noise,test_noise = sample_wise_perturbation(args, logger, num_classes, fingerprint_path, noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise,random_test_noise, device, ENV)
             
    elif args.attack_type == 'EM' or args.attack_type == 'EM-Pseudo':
        noise,test_noise = sample_wise_perturbation_min_min(args, logger, num_classes, fingerprint_path, noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise,random_test_noise, device, ENV)      
    
    else:
        raise('Not implemented yet')
    
    torch.save(noise, os.path.join(args.exp_name, 'perturbation.pt'))
    logger.info(noise)
    logger.info(noise.shape)
    logger.info('Noise saved at %s' % (os.path.join(args.exp_name, 'perturbation.pt')))

    torch.save(test_noise, os.path.join(args.exp_name, 'test_perturbation.pt'))
    logger.info(test_noise)
    logger.info(test_noise.shape)
    logger.info('Test noise saved at %s' % (os.path.join(args.exp_name, 'test_perturbation.pt')))  
    return

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f"Execution time: {(end - start) / 60:.2f} minutes")