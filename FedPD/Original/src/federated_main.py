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
from models import Model1
from utils import get_dataset, average_weights, exp_details, setup_seed

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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

    local_model, model, alpha, local_theta = [], [], [], []

    if args.model == 'Model1':
        global_model = Model1(args=args)
    else:
        exit('Error: unrecognized model')

    for idx in range(args.num_users):
        local_model.append(LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger))
        model.append(global_model) 
        weight = model[idx].state_dict()
        temp = {}
        temp_theta = {}
        for key in weight.keys():
            temp[key] = torch.zeros_like(weight[key])
            temp_theta[key] = temp[key] + weight[key]
        alpha.append(temp)
        local_theta.append(temp_theta)
        model[idx].to(args.device)
    
    global_model.to(args.device)
    global_model.train()
    print(global_model)

    global_weights = global_model.state_dict()

    theta = copy.deepcopy(global_weights)  

    test_acc, test_loss = [], []
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)): 

        local_losses, local_sum = [], []   
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:        
            loss, lsum, model[idx], alpha[idx] = local_model[idx].update_weights(model[idx], global_round=epoch, alpha=alpha[idx], theta=local_theta[idx])
            local_theta[idx] = copy.deepcopy(lsum)
            local_losses.append(copy.deepcopy(loss))
            local_sum.append(copy.deepcopy(lsum))

        a = random.random()
        if a > args.prob:
            theta = average_weights(local_sum)
            for idx in idxs_users:
                local_theta[idx] = copy.deepcopy(theta)
        else:
            print('----------------jumped Communication------------------')

        global_model.load_state_dict(theta)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        # sys.exit(0)
        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            acc, loss = local_model[c].inference(model[c])
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        test_acc_1, test_loss_1 = test_inference(args, global_model, test_dataset)
        print('\ntest accuracy:{:.2f}%\n'.format(100*test_acc_1))
        test_acc.append(test_acc_1)
        test_loss.append(test_loss_1)

        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc[-1]))

    file_name = '../save/{}_{}.pkl'.format(args.file_name, args.seed)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, test_acc], f)


    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    # PLOTTING (optional)
    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_user{}_globalepoch{}_C[{}]_iid[{}]_E[{}]_B[{}]_eta{}_mu{}_loss.png'.
                format(args.dataset, args.model, args.num_users, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.lr, args.rho))
    #
    # # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_user{}_globalepoch{}_C[{}]_iid[{}]_E[{}]_B[{}]_eta{}_mu{}_train_acc.png'.
                format(args.dataset, args.model, args.num_users, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.lr, args.rho))
    # Plot test Accuracy vs Communication rounds
    plt.figure()
    plt.title('test_acc vs Communication rounds')
    plt.plot(range(len(test_acc)), test_acc, color='r')
    plt.ylabel('test_acc')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_user{}_globalepoch{}_C[{}]_iid[{}]_E[{}]_B[{}]_eta{}_mu{}_test_acc.png'.
                format(args.dataset, args.model, args.num_users, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.lr, args.rho))
    
    # Plot test loss vs Communication rounds
    plt.figure()
    plt.title('test_loss vs Communication rounds')
    plt.plot(range(len(test_loss)), test_loss, color='r')
    plt.ylabel('test_loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_user{}_globalepoch{}_C[{}]_iid[{}]_E[{}]_B[{}]_eta{}_mu{}_test_loss.png'.
                format(args.dataset, args.model, args.num_users, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.lr, args.rho))
