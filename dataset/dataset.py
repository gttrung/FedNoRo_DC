import numpy as np
import torch
# import torchvision.transforms as transforms
from torchvision import datasets, transforms
from .all_datasets import isic2019, ICH

from util.sampling import iid_sampling, non_iid_dirichlet_sampling


# def get_dataset(args):
#     if args.dataset == "isic2019":
#         root = "your path"
#         args.n_classes = 8

#         normalize = transforms.Normalize([0.485, 0.456, 0.406],
#                                          [0.229, 0.224, 0.225])
#         train_transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ])
#         val_transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             normalize,
#         ])

#         train_dataset = isic2019(root, "train", train_transform)
#         test_dataset = isic2019(root, "test", val_transform)


#     elif args.dataset == "ICH":
#         root = "your path"
#         args.n_classes = 5

#         normalize = transforms.Normalize([0.485, 0.456, 0.406],
#                                          [0.229, 0.224, 0.225])
#         train_transform = transforms.Compose([
#             transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ])
#         val_transform = transforms.Compose([
#             transforms.ToTensor(),
#             normalize,
#         ])

#         train_dataset = ICH(root, "train", train_transform)
#         test_dataset = ICH(root, "test", val_transform)

#     else:
#         exit("Error: unrecognized dataset")

#     n_train = len(train_dataset)
#     y_train = np.array(train_dataset.targets)
#     assert n_train == len(y_train)

#     if args.iid:
#         dict_users = iid_sampling(n_train, args.n_clients, args.seed)
#     else:
#         dict_users = non_iid_dirichlet_sampling(y_train, args.n_classes, args.non_iid_prob_class, args.n_clients, seed=100, alpha_dirichlet=args.alpha_dirichlet)

#     # check
#     assert len(dict_users.keys()) == args.n_clients
#     items = []
#     for key in dict_users.keys():
#         items += list(dict_users[key])
#     assert len(items) == len(set(items)) == len(y_train)

#     print("### Datasets are ready ###")
    
#     return train_dataset, test_dataset, dict_users
def get_dataset(args):
    # args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if args.dataset == 'cifar10':
        data_path = './data/cifar10'
        args.n_classes = 10
        args.model = 'Resnet18'
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        dataset_train = datasets.CIFAR10(data_path, train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR10(data_path, train=False, download=True, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)
    elif args.dataset == 'cifar100':
        data_path = './data/cifar100'
        args.n_classes = 100
        args.model = 'Resnet34'
        
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                 std=[0.267, 0.256, 0.276])],
        )
        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                 std=[0.267, 0.256, 0.276])],
        )
        dataset_train = datasets.CIFAR100(data_path, train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR100(data_path, train=False, download=True, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)

    else:
        exit('Error: unrecognized dataset')

    if args.iid:
        dict_users = iid_sampling(n_train, args.n_clients, args.seed)
        return dataset_train, dataset_test, dict_users
        
    else:
        dict_users = non_iid_dirichlet_sampling(y_train, args.n_classes, args.non_iid_prob_class, args.n_clients, args.seed, args.alpha_dirichlet)
        return dataset_train, dataset_test, dict_users


