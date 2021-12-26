import copy
import torch
import torch.nn as nn
import numpy as np
import random
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal

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
    elif args.unequal == 1:
        user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
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

def generateLocalEpochs(size, args):
  ''' Method generates list of epochs for selected clients
  to replicate system heteroggeneity

  Params:
    threshold: threshold of clients to have fewer than E epochs
    size:       total size of the list
    max_epochs: maximum value for local epochs
  
  Returns:
    List of size epochs for each Client Update

  '''

  # if threshold is 0 then each client runs for E epochs
  if args.threshold == 0:
      return np.array([args.epochs]*size)
  else:
    # get the number of clients to have fewer than E epochs
    stragglers = int((args.threshold/100) * size)

    # generate random uniform epochs of heterogenous size between 1 and E
    epoch_list = np.random.randint(1, args.local_ep, stragglers)

    # the rest of the clients will have E epochs
    remaining_size = size - stragglers
    rem_list = [args.local_ep]*remaining_size

    epoch_list = np.append(epoch_list, rem_list, axis=0)
    
    # shuffle the list and return
    np.random.shuffle(epoch_list)

    return epoch_list

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    elif args.unequal == 1:
        print('    Non-IID: unequal')
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
