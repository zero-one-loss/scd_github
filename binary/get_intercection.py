import numpy as np
import pickle
import sys
sys.path.append('..')
from core.bnn import BNN
from tools import args, save_checkpoint, print_title, load_data
import time
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from core.sim import SimModel
from core.cnn_ensemble import CNNVote




if __name__ == '__main__':


    models = [
        'cifar10_scd01mlp_100_br02_nr075_ni1000_i1.pkl',
        'cifar10_mlp_100.pkl',
        'cifar10_mlpbnn_approx',
        'cifar10_rf_100.pkl',
        'cifar10_scdcemlp_100_br02_h20_nr075_ni1000_i1_0.pkl',
        'cifar10_scdcemlpbnn_100_br02_h20_nr075_ni10000_i1_0.pkl',
        'cifar10_scd01mlpbnn_100_br02_h20_nr075_ni10000_i1_0.pkl',
        'cifar10_lenet_100.pkl',
    ]
    np.random.seed(2018)

    train_data, test_data, train_label, test_label = load_data('cifar10', 2)

    correct_index = []

    for model in models:
        print(model)
        if 'approx' in model:
            scd = BNN(['checkpoints/%s_%d.h5' % (model, i) for i in range(100)])
        elif 'simclr' in model:
            scd = SimModel(sim_path='checkpoints/128_0.5_200_512_500_model.pth',
                           scd_path='checkpoints/%s' % model)

        elif 'resnet' in model:
                scd = CNNVote('checkpoints/%s' % model, 10)
        else:
            with open('checkpoints/%s' % model, 'rb') as f:
                scd = pickle.load(f)
                # scd.best_model.status = 'sign'
                # for i in range(len(scd.models)):
                #     scd.models[i].status = 'sign'
        if 'lenet' in model:
            yp = scd.predict(test_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2)).astype(np.float32))
        else:
            yp = scd.predict(test_data)
        del scd
        correct_index.append((yp == test_label).astype(np.int8))

    correct_index = np.stack(correct_index, axis=1).sum(axis=1) // len(models)
    correct_index = np.nonzero(correct_index)[0]
    np.save('cifar10_correct_index.npy', correct_index)
    # yp = scd.predict(test_data)
    # print('clean accuracy: ', accuracy_score(test_label, yp))
    # acc = []
    # for i in range(10):
    #     print('%d run on epsilon %.3f:' % (i+1, args.epsilon))
    # # if scd_args.normal_noise:
    #     noise = np.random.normal(0, 1, size=test_data.shape)
    #     noisy = np.clip((test_data + noise * args.epsilon), 0, 1)
    #     yp = scd.predict(noisy)
    #     temp_acc = accuracy_score(test_label, yp)
    #     print('noise accuracy: ', temp_acc)
    #     acc.append(temp_acc)
    # print('Average accuracy: ', np.mean(acc))
