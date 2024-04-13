import copy
import random
import os
import sys
import shutil
import numpy as np
import pandas as pd
import logging
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F

from sklearn.metrics import balanced_accuracy_score


# def add_noise(args, y_train, dict_users):
#     np.random.seed(args.seed)
#     gamma_s = np.array([0.] * args.n_clients)
#     gamma_s[:int(args.level_n_system*args.n_clients)] = 1.
#     np.random.shuffle(gamma_s)
#     gamma_c_initial = np.random.rand(args.n_clients)
#     gamma_c_initial = (args.level_n_upperb - args.level_n_lowerb) * \
#         gamma_c_initial + args.level_n_lowerb
#     gamma_c = gamma_s * gamma_c_initial
#     y_train_noisy = copy.deepcopy(y_train)

#     real_noise_level = np.zeros(args.n_clients)
#     for i in np.where(gamma_c > 0)[0]:
#         sample_idx = np.array(list(dict_users[i]))
#         prob = np.random.rand(len(sample_idx))
#         noisy_idx = np.where(prob <= gamma_c[i])[0]
#         if args.n_type == "random":
#             y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(
#                 0, args.n_classes, len(noisy_idx))
#         elif args.n_type == "pairwise":
#             y_train_noisy[sample_idx[noisy_idx]] = (y_train[sample_idx[noisy_idx]] - 1) % args.n_classes
#         noise_ratio = np.mean(
#             y_train[sample_idx] != y_train_noisy[sample_idx])
#         logging.info("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (
#             i, gamma_c[i], gamma_c[i] * 0.9, noise_ratio))
#         real_noise_level[i] = noise_ratio

#     return (y_train_noisy, gamma_s, real_noise_level)

def add_noise(args, y_train, dict_users):
    np.random.seed(args.seed)

    gamma_s = np.random.binomial(1, args.level_n_system, args.n_clients)
         
    gamma_c_initial = np.random.rand(args.n_clients)
    
    gamma_c_initial = (1 - args.level_n_lowerb) * gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_s * gamma_c_initial
    gamma_c_mix = np.zeros_like(gamma_c,dtype=int)
    # gamma_c_mix[gamma_c > 0] = np.random.binomial(1, args.level_system_mix, np.sum(gamma_c > 0))+1
    # print(gamma_c_mix,'\n',gamma_c)
    # print(np.where(gamma_c_mix != 0)[0],'\n',np.where(gamma_c != 0)[0],'\n',np.allclose(np.where(gamma_c_mix != 0)[0],np.where(gamma_c != 0)[0]))
    y_train_noisy = copy.deepcopy(y_train)
    real_noise_ratio = np.zeros(args.n_clients)
    noise_types = ['none','symmetric', 'pairflip']

    for i in np.where(gamma_c > 0)[0]:
        sample_idx = np.array(list(dict_users[i]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= gamma_c[i])[0]
        if args.n_type == 'symmetric':
            y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, args.n_classes, len(noisy_idx))
        elif args.n_type == 'pairflip':
            y_train_noisy[sample_idx[noisy_idx]] = (y_train[sample_idx[noisy_idx]] - 1) % args.n_classes
        elif args.n_type == "mix":
            if gamma_c_mix[i] == 1:
                y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, args.n_classes, len(noisy_idx))
            elif gamma_c_mix[i] == 2:
                y_train_noisy[sample_idx[noisy_idx]] = (y_train[sample_idx[noisy_idx]] - 1) % args.n_classes
        real_noise_ratio[i] = np.mean(y_train[sample_idx] != y_train_noisy[sample_idx])
        if args.n_type == 'mix':
            print("Client %d, init noise level: %.4f ,real noise ratio: %.4f, noise type: %s" % (i, gamma_c[i], real_noise_ratio[i], noise_types[gamma_c_mix[i]]))
        else:
            print("Client %d, init noise level: %.4f ,real noise ratio: %.4f" % (i, gamma_c[i], real_noise_ratio[i]))
    return y_train_noisy, gamma_s, real_noise_ratio

def add_noise_FedDC(args, y_train, dict_users, new_users):
    np.random.seed(args.seed)

    gamma_s = np.random.binomial(1, args.level_n_system, args.n_clients - args.n_new_clients)
    
    if args.n_new_clients !=0 :
      gamma_s_new = np.random.binomial(1, args.level_n_new_system, args.n_new_clients)
      gamma_s = np.hstack((gamma_s, gamma_s_new))
         
    gamma_c_initial = np.random.rand(args.n_clients)
    
    gamma_c_initial = (1 - args.level_n_lowerb) * gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_s * gamma_c_initial
    y_train_noisy = copy.deepcopy(y_train)
    real_noise_level = np.zeros(args.n_clients)

    for i in np.where(gamma_c > 0)[0]:
        if i not in list(new_users.keys()):
            sample_idx = np.array(list(dict_users[i]))
        else:
            sample_idx = np.array(list(new_users[i]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= gamma_c[i])[0]
        y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, args.n_classes, len(noisy_idx))
        real_noise_level[i] += np.mean(y_train[sample_idx] != y_train_noisy[sample_idx])
        if i < args.n_clients - args.n_new_clients:
            print("Client %d, init noise level: %.4f (%.4f) ,real noise ratio: %.4f" % (i, gamma_c[i], gamma_c[i] * 0.9, real_noise_level[i]))
        else:
            print("New client %d, init noise level: %.4f (%.4f) ,real noise ratio: %.4f" % (i, gamma_c[i], gamma_c[i] * 0.9, real_noise_level[i]))
    return y_train_noisy, gamma_s, real_noise_level

def sigmoid_rampup(current, begin, end):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    current = np.clip(current, begin, end)
    phase = 1.0 - (current-begin) / (end-begin)
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(rnd, begin, end):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return sigmoid_rampup(rnd, begin, end)


def get_output(loader, net, args, softmax=False, criterion=None):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            labels = labels.long()
            if softmax == True:
                outputs = net(images)
                outputs = F.softmax(outputs, dim=1)
            else:
                outputs = net(images)
            if criterion is not None:
                loss = criterion(outputs, labels)
            if i == 0:
                output_whole = np.array(outputs.cpu())
                if criterion is not None:
                    loss_whole = np.array(loss.cpu())
            else:
                output_whole = np.concatenate(
                    (output_whole, outputs.cpu()), axis=0)
                if criterion is not None:
                    loss_whole = np.concatenate(
                        (loss_whole, loss.cpu()), axis=0)
    if criterion is not None:
        return output_whole, loss_whole
    else:
        return output_whole


def get_output_and_label(loader, net, args):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            labels = labels.long()

            outputs = net(images)
            outputs = F.softmax(outputs, dim=1)

            if i == 0:
                output_whole = np.array(outputs.cpu())
                label_whole = np.array(labels.cpu())
            else:
                output_whole = np.concatenate(
                    (output_whole, outputs.cpu()), axis=0)
                label_whole = np.concatenate(
                    (label_whole, labels.cpu()), axis=0)

    return output_whole, label_whole


def cal_training_acc(prediction, noisy_labels, true_labels):
    prediction = np.array(prediction)
    noisy_labels = np.array(noisy_labels)
    true_labels = np.array(true_labels)

    acc_noisy = balanced_accuracy_score(noisy_labels, prediction)
    acc_true = balanced_accuracy_score(true_labels, prediction)

    return acc_noisy, acc_true


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_output_files(args):
    outputs_dir = 'outputs_' + str(args.dataset) + '_' + str(
        args.level_n_system) + '_' + str(args.level_n_lowerb) + '_' + str(args.level_n_upperb)
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)
    exp_dir = os.path.join(outputs_dir, args.exp + '_' +
                           str(args.level_n_system) + '_' + str(args.level_n_lowerb) + '_' +
                           str(args.level_n_upperb) + '_' + str(args.local_ep))
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    models_dir = os.path.join(exp_dir, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    logs_dir = os.path.join(exp_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    tensorboard_dir = os.path.join(exp_dir, 'tensorboard')
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    code_dir = os.path.join(exp_dir, 'code')
    if os.path.exists(code_dir):
        shutil.rmtree(code_dir)
    # shutil.copytree('.', code_dir, ignore=shutil.ignore_patterns('.git'))

    logging.basicConfig(filename=logs_dir+'/logs.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    writer = SummaryWriter(tensorboard_dir)
    return writer, models_dir
