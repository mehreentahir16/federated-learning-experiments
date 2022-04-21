import os
import copy
import time
import pickle
import random
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import mnist_cnn
from utils import get_dataset, average_weights, exp_details, setup_seed, generateLocalEpochs

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def client_sampling(n, m, weights=None, with_replace=False):
    pk = None
    if weights:
        total_weights = np.sum(np.asarray(weights))
        pk = [w * 1.0 / total_weights for w in weights]

    return np.random.choice(range(n), m, replace=with_replace, p=pk)

if __name__ == '__main__':

    path_project = os.path.abspath('.')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    setup_seed(args.seed)
    print('random seed =', args.seed)

    args.device = torch.device("cpu")

    train_dataset, test_dataset, user_groups = get_dataset(args)

    if args.model == 'mnist_cnn':
        global_model = mnist_cnn(args=args)
    else:
        exit('Error: unrecognized model')

    global_model.to(args.device)
    global_model.train()
    print(global_model)

    global_weights = global_model.state_dict()

    test_acc, test_loss = [], []
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    q=0
    sampling='uniform'

    # client weights by total samples
    p_k = None
    if sampling == 'weighted':
        p_k = [len(user_groups[c]) for c in user_groups] if args.dataset else [len(user_groups[c]['train_dataset']) for c in user_groups]
    
    start_time = time.time()
    users_id = list(user_groups.keys())

    for epoch in tqdm(range(args.epochs)):
        deltas, hs, local_loss = [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
 
        idxs_users = client_sampling(args.num_users, m, weights=p_k, with_replace=False)

        global_weights = global_model.state_dict()

        if args.threshold==0:
            for idx in idxs_users:
                key = users_id[idx]
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[key], logger=logger, q=q)
                delta_k, h_k, loss  = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
                deltas.append(copy.deepcopy(delta_k))
                hs.append(copy.deepcopy(h_k))
                local_loss.append(copy.deepcopy(loss))
        else:
            heterogenous_epoch_list = generateLocalEpochs(size=m, args=args)
            heterogenous_epoch_list = np.array(heterogenous_epoch_list)

            stragglers_indices = np.argwhere(heterogenous_epoch_list < args.local_ep)
            # for index in stragglers_indices:
            #     time.sleep(random.uniform(5, 50))

            idxs_active = np.delete(idxs_users, stragglers_indices)
            
            for idx in idxs_active:
                key = users_id[idx]
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[key], logger=logger, q=q)
                delta_k, h_k, loss  = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
                deltas.append(copy.deepcopy(delta_k))
                hs.append(copy.deepcopy(h_k))
                local_loss.append(copy.deepcopy(loss))

        # Perform qFedAvg
        h_sum = copy.deepcopy(hs[0])
        delta_sum = copy.deepcopy(deltas[0])

        for k in h_sum.keys():
            for i in range(1, len(hs)):
                h_sum[k] += hs[i][k]
                delta_sum[k] += deltas[i][k]

        new_weights = {}
        for k in delta_sum.keys():
            for i in range(len(deltas)):
                new_weights[k] = delta_sum[k] / h_sum[k]

        # Updating global model weights
        for k in global_weights.keys():
            global_weights[k] -= new_weights[k]
        # move the updated weights to our model state dict
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_loss) / len(local_loss)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            
        test_acc_1, test_loss_1 = test_inference(args, global_model, test_dataset)
        print('\ntest accuracy:{:.2f}%\n'.format(100*test_acc_1))
        test_acc.append(test_acc_1)
        test_loss.append(test_loss_1)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc[-1]))

    file_name = '../save/{}_{}_iid[{}]_E[{}].pkl'.format(args.file_name, args.seed, args.iid, args.epochs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, test_acc], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING 
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/qfed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss_P[{}].png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.threshold))
    
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/qfed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc_P[{}].png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.threshold))

    # Plot test Accuracy vs Communication rounds
    plt.figure()
    plt.title('test_acc vs Communication rounds')
    plt.plot(range(len(test_acc)), test_acc, color='r')
    plt.ylabel('test_acc')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/qfed_{}_{}_user{}_globalepoch{}_C[{}]_iid[{}]_E[{}]_B[{}]_eta{}_P[{}]_test_acc.png'.
                format(args.dataset, args.model, args.num_users, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.lr, args.threshold))
    
    # Plot test loss vs Communication rounds
    plt.figure()
    plt.title('test_loss vs Communication rounds')
    plt.plot(range(len(test_loss)), test_loss, color='r')
    plt.ylabel('test_loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/qfed_{}_{}_user{}_globalepoch{}_C[{}]_iid[{}]_E[{}]_B[{}]_eta{}_P[{}]_test_loss.png'.
                format(args.dataset, args.model, args.num_users, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.lr, args.threshold))
