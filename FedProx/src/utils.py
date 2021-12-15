import copy
import torch
import random
import numpy as np
import random
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_dataset(args):

    data_dir = '../data/mnist/'

    apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)


    if args.iid:
        user_groups = mnist_iid(train_dataset, args.num_users)
    else:
        user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    dataset            : {args.dataset}')
    print(f'    Num of users       : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Learning  Rate     : {args.lr}')
    print(f'    rho                : {args.rho}')
    print(f'    Local Epochs       : {args.local_ep}')
    print(f'    Local Batch size   : {args.local_bs}\n')
    return
