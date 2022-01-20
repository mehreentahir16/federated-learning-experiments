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


if __name__ == '__main__':
    start_time = time.time()

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

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)

        newMedian = max(int(0.01 * m), 1)
    
        c = 0
 
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for i in range(len(idxs_users)):
            if c == newMedian:
                break
            c += 1

        # heterogenous_epoch_list = generateLocalEpochs(size=m, args=args)
        # heterogenous_epoch_list = np.array(heterogenous_epoch_list)

        # stragglers_indices = np.argwhere(heterogenous_epoch_list < args.local_ep)

        # idxs_active = np.delete(idxs_users, stragglers_indices)

            for idx in idxs_users:
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
                weights, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            # local_weights.append(copy.deepcopy(w))
            # local_losses.append(copy.deepcopy(loss))
                for k in weights.keys():
                    t = torch.Tensor(weights[k].shape)
                    t.fill_(0.1)
                    weights[k] = t     

                local_weights.append(copy.deepcopy(weights))
                local_losses.append(copy.deepcopy(loss))

        for k in range(newMedian, len(idxs_users)):
            for idx in idxs_users:
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
                weights, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
      
                local_weights.append(copy.deepcopy(weights))
                local_losses.append(copy.deepcopy(loss))

        ix = 0
        weights = []
        new_w = local_weights
        for k in local_weights[ix].keys():
            l1 = []
            for i in range(len(local_weights)):

                # Fixed Precision
                x1 = local_weights[i][k] * (10**3)
          
                # Flattening weight tensors to lists
                f1 = torch.flatten(x1)
          
                # Converting values to int for our streaming algo 
                f1 = list(map(int, f1))
          
                l1.append(f1)
                # print(len(f1))
          
            # Computing Median
            j = 0
            gl = []
            while j < len(l1[0]):
                nl = []
                for i in range(len(l1)):
                    nl.append(l1[i][j])
                m = np.median(nl)
                gl.append(m)
                j += 1

      
            for i in range(len(gl)):
                gl[i] /= (10**3)
      
            # print(len(gl))
      
            newar = np.asarray(gl)
            newar = np.reshape(gl, local_weights[0][k].size())
            tens = torch.from_numpy(newar)
            weights.append(tens)

            ix += 1

        k = 0
        coun = 0
        for key, value in global_weights.items():
            global_weights[key] = weights[k]
            k += 1
            coun += 1

        # global_weights = average_weights(local_weights)

        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
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

    file_name = '../save/{}_{}.pkl'.format(args.file_name, args.seed)

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
    plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss_P[{}].png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.threshold))
    
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc_P[{}].png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.threshold))

    # Plot test Accuracy vs Communication rounds
    plt.figure()
    plt.title('test_acc vs Communication rounds')
    plt.plot(range(len(test_acc)), test_acc, color='r')
    plt.ylabel('test_acc')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/new/fed_{}_{}_user{}_globalepoch{}_C[{}]_iid[{}]_E[{}]_B[{}]_eta{}_mu{}_P[{}]_test_acc.png'.
                format(args.dataset, args.model, args.num_users, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.lr, args.rho, args.threshold))
    
    # Plot test loss vs Communication rounds
    plt.figure()
    plt.title('test_loss vs Communication rounds')
    plt.plot(range(len(test_loss)), test_loss, color='r')
    plt.ylabel('test_loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/new/fed_{}_{}_user{}_globalepoch{}_C[{}]_iid[{}]_E[{}]_B[{}]_eta{}_mu{}_P[{}]_test_loss.png'.
                format(args.dataset, args.model, args.num_users, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.lr, args.rho, args.threshold))
