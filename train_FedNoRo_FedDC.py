import os
import copy
import logging
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from collections import Counter

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from util.options import args_parser
from util.local_training import LocalUpdate, globaltest, globaltest_base
from util.FedAvg import FedAvg, DaAgg
from util.utils import add_noise, add_noise_FedDC, set_seed, set_output_files, get_output, get_current_consistency_weight
from util.dynamic import separate_users, merge_users

from dataset.dataset import get_dataset
from model.build_model import build_model
np.set_printoptions(threshold=np.inf)

"""
Major framework of noise FL
"""

if __name__ == '__main__':
    args = args_parser()
    args.n_clients = args.n_clients
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------ deterministic or not ------------------------------
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        set_seed(args.seed)

    # ------------------------------ output files ------------------------------
    writer, models_dir = set_output_files(args)

    # ------------------------------ dataset ------------------------------
    
    dataset_train, dataset_test, dict_users = get_dataset(args)
    
    # ------------------------------ separate users ------------------------------
    
    dict_users, new_users = separate_users(args, dict_users)

    # ----------------------------------------------------------------------------
    logging.info(
        f"train: {Counter(dataset_train.targets)}, total: {len(dataset_train.targets)}")
    logging.info(
        f"test: {Counter(dataset_test.targets)}, total: {len(dataset_test.targets)}")
    logging.info(
        f'Number of new clients: {len(new_users)}')
    
    # --------------------------- Add Noise ---------------------------
    y_train = np.array(dataset_train.targets)
    y_train_noisy, gamma_s, real_noise_level = add_noise_FedDC(
        args, y_train, dict_users, new_users)
    
    args.n_clients = args.n_clients - args.n_new_clients
    args.frac1 = 1/args.n_clients
    num_user_new = args.n_clients
    num_user_old = args.n_clients
    dataset_train.targets = y_train_noisy

    # --------------------------- Build Models ---------------------------
    netglob = build_model(args)
    user_id = list(range(args.n_clients))
    trainer_locals = []
    # for id in user_id:
    #     print(id, end=' ')
    #     trainer_locals.append(LocalUpdate(
    #         args, id, copy.deepcopy(dataset_train), dict_users[id]))

    # ------------------------------ begin training ------------------------------
    set_seed(args.seed)
    logging.info("\n ---------------------begin training---------------------")
    best_performance = 0.

    # ------------------------ Stage 1: warm up ------------------------ 
    if args.warm:
        for rnd in range(args.s1):
            w_locals, loss_locals = [], []
            if rnd == args.joining_round[0]:
                
                new_clients = np.arange(args.n_clients, args.n_clients + args.n_new_clients*args.stage_ratio).astype(int)
                dict_users = merge_users(dict_users, new_users, args, stage = 1)
                args.n_clients += len(new_clients)
                num_user_new = args.n_clients
                new_noisy_clients = []
                new_clean_clients = []
                user_id = list(range(args.n_clients))
                compute_loss_thresh = True
            
            if rnd == 0:
                loss_thresh = 0

            for idx in user_id:  # training over the subset
                # local = trainer_locals[idx]
                local = LocalUpdate(args, idx, copy.deepcopy(dataset_train), dict_users[idx])
                w_local, loss_local, loss_FedDC = local.train_LA(
                    net=copy.deepcopy(netglob).to(args.device), writer=writer)
                
                if args.method == 'loss_thresh':
                    if num_user_old < num_user_new:
                        if idx in new_clients:
                            if compute_loss_thresh:
                                loss_thresh /= num_user_old
                                print(loss_thresh)
                                compute_loss_thresh = False
                            num_user_old += 1
                            if loss_local >= loss_thresh:
                                print(f'Noisy client {idx}:{loss_local}')
                                new_noisy_clients.append(idx)
                            else:
                                print(f'Clean client {idx}:{loss_local}')
                                new_clean_clients.append(idx)
                        else:
                            loss_thresh += loss_local

                # store every updated model
                w_locals.append(copy.deepcopy(w_local))
                loss_locals.append(copy.deepcopy(loss_local))
            # if args.method == 'loss_thresh':
            #     # if rnd == 0:
            #     if rnd == args.joining_round[0]:
            #         loss_thresh /= args.n_clients
            w_locals_last = copy.deepcopy(w_locals)
            dict_len = [len(dict_users[idx]) for idx in user_id]
            w_glob_fl = FedAvg(w_locals, dict_len)
            netglob.load_state_dict(copy.deepcopy(w_glob_fl))

            pred = globaltest(copy.deepcopy(netglob).to(
                args.device), dataset_test, args)
            acc_base = globaltest_base(copy.deepcopy(netglob).to(
                args.device), dataset_test, args)
            acc = accuracy_score(dataset_test.targets, pred)
            bacc = balanced_accuracy_score(dataset_test.targets, pred)
            cm = confusion_matrix(dataset_test.targets, pred)
            logging.info(
                "******** round: %d, acc: %.4f, acc_base :%.4f, bacc: %.4f ********" % (rnd, acc, acc_base, bacc))
            logging.info(cm)
            writer.add_scalar(f'test/acc', acc, rnd)
            writer.add_scalar(f'test/bacc', bacc, rnd)

            # save model
            if bacc > best_performance:
                best_performance = bacc
            logging.info(f'best bacc: {best_performance}, now bacc: {bacc}')
            logging.info('\n')
        torch.save(netglob.state_dict(),  models_dir +
                   f'/stage1_model_{rnd}.pt')

    #  ------------------------ client selection ------------------------
    model_path = f"outputs_{args.dataset}_{args.level_n_system}_{args.level_n_lowerb}_{args.level_n_upperb}/{args.exp}_{args.level_n_system}_{args.level_n_lowerb}_{args.level_n_upperb}_{args.local_ep}/models/stage1_model_{args.s1-1}.pt"
    logging.info(
        f"********************** load model from: {model_path} **********************")
    netglob.load_state_dict(torch.load(model_path))
    loader = DataLoader(dataset=dataset_train, batch_size=32,
                        shuffle=False, num_workers=2)
    criterion = nn.CrossEntropyLoss(reduction='none')
    local_output, loss = get_output(
        loader, netglob.to(args.device), args, False, criterion)
    metrics = np.zeros((args.n_clients, args.n_classes)).astype("float")
    num = np.zeros((args.n_clients, args.n_classes)).astype("float")
    for id in range(args.n_clients):
        idxs = dict_users[id]
        for idx in idxs:
            c = dataset_train.targets[idx]
            num[id, c] += 1
            metrics[id, c] += loss[idx]
    metrics = metrics / num
    for i in range(metrics.shape[0]):
        for j in range(metrics.shape[1]):
            if np.isnan(metrics[i, j]):
                metrics[i, j] = np.nanmin(metrics[:, j])
    for j in range(metrics.shape[1]):
        metrics[:, j] = (metrics[:, j]-metrics[:, j].min()) / \
            (metrics[:, j].max()-metrics[:, j].min())
    logging.info("metrics:")
    logging.info(metrics)

    vote = []
    for i in range(9):
        gmm = GaussianMixture(n_components=2, random_state=i).fit(metrics)
        gmm_pred = gmm.predict(metrics)
        noisy_clients = np.where(gmm_pred == np.argmax(gmm.means_.sum(1)))[0]
        noisy_clients = set(list(noisy_clients))
        vote.append(noisy_clients)
    cnt = []
    for i in vote:
        cnt.append(vote.count(i))
    noisy_clients = list(vote[cnt.index(max(cnt))])

    if args.method == 'loss_thresh':
        if len(new_noisy_clients) > 0:
            for idx in new_noisy_clients:
                if idx not in noisy_clients:
                    noisy_clients.append(idx)

    logging.info(
        f"selected noisy clients: {noisy_clients},\n real noisy clients: {np.where(gamma_s>0.)[0]}")
    logging.info(f"new selected noisy clients: {new_noisy_clients}")
    clean_clients = list(set(user_id) - set(noisy_clients))

    logging.info(f"selected clean clients: {clean_clients}")
    logging.info(f"new selected clean clients: {new_clean_clients}")

    # ------------------------ Stage 2: ------------------------ 
    BACC = []
    for rnd in range(args.s1, args.rounds):
        w_locals, loss_locals = [], []
        weight_kd = get_current_consistency_weight(
            rnd, args.begin, args.end) * args.a
        writer.add_scalar(f'train/w_kd', weight_kd, rnd)

        if rnd == args.joining_round[1]:
                
                new_clients = np.arange(args.n_clients, args.n_clients + args.n_new_clients*args.stage_ratio).astype(int)
                dict_users = merge_users(dict_users, new_users, args, stage = 1)
                args.n_clients += len(new_clients)
                num_user_new = args.n_clients
                user_id = list(range(args.n_clients))

        for idx in user_id:  # training over the subset
            # local = trainer_locals[idx]
            local = LocalUpdate(args, idx, copy.deepcopy(dataset_train), dict_users[idx])
            if rnd == args.joining_round[1] and idx in new_clients:
                w_local, loss_local, loss_FedDC = local.train_LA(
                    net=copy.deepcopy(netglob).to(args.device), writer=writer)
                if args.method == 'loss_thresh':
                    if loss_local >= loss_thresh:
                        noisy_clients.append(idx)
                        logging.info(f"new noisy client: {idx}")
                    else:
                        clean_clients.append(idx)
                        logging.info(f"new clean client: {idx}")
            elif idx in clean_clients:
                w_local, loss_local, loss_FedDC = local.train_LA(
                    net=copy.deepcopy(netglob).to(args.device), writer=writer)
            elif idx in noisy_clients:
                w_local, loss_local = local.train_FedNoRo(
                    student_net=copy.deepcopy(netglob).to(args.device), teacher_net=copy.deepcopy(netglob).to(args.device), writer=writer, weight_kd=weight_kd)
            # store every updated model
            w_locals.append(copy.deepcopy(w_local))
            loss_locals.append(copy.deepcopy(loss_local))
            assert len(w_locals) == len(loss_locals) == idx+1

        dict_len = [len(dict_users[idx]) for idx in user_id]
        w_glob_fl = DaAgg(
            w_locals, dict_len, clean_clients, noisy_clients)
        netglob.load_state_dict(copy.deepcopy(w_glob_fl))

        pred = globaltest(copy.deepcopy(netglob).to(
            args.device), dataset_test, args)
        acc = accuracy_score(dataset_test.targets, pred)
        bacc = balanced_accuracy_score(dataset_test.targets, pred)
        cm = confusion_matrix(dataset_test.targets, pred)
        logging.info(
            "******** round: %d, acc: %.4f, bacc: %.4f ********" % (rnd, acc, bacc))
        logging.info(cm)
        writer.add_scalar(f'test/acc', acc, rnd)
        writer.add_scalar(f'test/bacc', bacc, rnd)
        BACC.append(bacc)

        # save model
        if bacc > best_performance:
            best_performance = bacc
        logging.info(f'best bacc: {best_performance}, now bacc: {bacc}')
        logging.info('\n')
    torch.save(netglob.state_dict(),  models_dir+'f/stage2_model_{rnd}.pth')

    BACC = np.array(BACC)
    logging.info("last:")
    logging.info(BACC[-10:].mean())
    logging.info("best:")
    logging.info(BACC.max())

    torch.cuda.empty_cache()    
