import logging
import numpy as np

import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from util.losses import LogitAdjust, LA_KD


def globaltest(net, test_dataset, args):
    net.eval()
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    pred = np.array([])
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            pred = np.concatenate([pred, predicted.detach().cpu().numpy()], axis=0)
    return pred

def globaltest_base(net, test_dataset, args):
    net.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            outputs = net(images)
            # outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    return acc

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return self.idxs[item], image, label

    def get_num_of_each_class(self, args):
        class_sum = np.array([0] * args.n_classes)
        for idx in self.idxs:
            label = self.dataset.targets[idx]
            class_sum[label] += 1
        return class_sum.tolist()



class LocalUpdate(object):
    def __init__(self, args, id, dataset, idxs):
        self.args = args
        self.id = id
        self.idxs = idxs
        self.local_dataset = DatasetSplit(dataset, idxs)
        self.class_num_list = self.local_dataset.get_num_of_each_class(self.args)
        # logging.info(
        #     f'client{id} each class num: {self.class_num_list}, total: {len(self.local_dataset)}')
        self.ldr_train = DataLoader(
            self.local_dataset, batch_size=self.args.batch_size, shuffle=True, 
            num_workers=2, drop_last=True)
        self.epoch = 0
        self.iter_num = 0
        self.lr = self.args.base_lr


    def train_LA(self, net, writer):
        net_glob = net.to(self.args.device)

        net.train()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)

        # train and update
        epoch_loss = []
        epoch_loss_FedDC = []
        ce_criterion = LogitAdjust(cls_num_list=self.class_num_list)
        for epoch in range(self.args.local_ep):
            batch_loss = []
            batch_loss_FedDC = []
            # based on cyclic learning rate foluma in O2U-Net paper
            if self.args.lr_cyclic:
                scale = (epoch % (self.args.local_ep//2)) / float(self.args.local_ep//2)
                adjust_lr = (1 - scale) * self.lr + scale * self.args.lr_min
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = adjust_lr

            for (_, images, labels) in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                self.optimizer.zero_grad()
                logits = net(images)
                loss = ce_criterion(logits, labels)
                if self.args.beta > 0:
                    if _ > 0:
                        w_diff = torch.tensor(0.).to(self.args.device)
                        for w, w_t in zip(net_glob.parameters(), net.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        w_diff = torch.sqrt(w_diff)
                        # loss += self.args.beta * mu * w_diff
                        loss_FedDC = loss + self.args.beta * w_diff
                

                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                batch_loss_FedDC.append(loss_FedDC.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())
            epoch_loss_FedDC.append(np.array(batch_loss_FedDC).mean())

        return net.state_dict(), np.array(epoch_loss).mean(), sum(epoch_loss_FedDC) / len(epoch_loss_FedDC)
    

    def train_FedNoRo(self, student_net, teacher_net, writer, weight_kd):
        student_net.train()
        teacher_net.eval()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            student_net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)

        # train and update
        epoch_loss = []
        criterion = LA_KD(cls_num_list=self.class_num_list)
        
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (img_idx, images, labels) in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                self.optimizer.zero_grad()
                logits = student_net(images)
                with torch.no_grad():
                    teacher_output = teacher_net(images)
                    soft_label = torch.softmax(teacher_output/0.8, dim=1)

                loss = criterion(logits, labels, soft_label, weight_kd)

                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return student_net.state_dict(), np.array(epoch_loss).mean()

    
  