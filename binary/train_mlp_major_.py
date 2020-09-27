import numpy as np
import pickle
import sys
sys.path.append('..')
from tools import args, save_checkpoint, print_title, load_data
import time
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier

if __name__ == '__main__':
    # Set Random Seed
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # print information
    et, vc = print_title()
    save = True if args.save else False
    normal_noise = True if args.normal_noise else False
    train, test, train_label, test_label = load_data(args.dataset, args.n_classes)
    random_patch = True if args.random_patch else False
    normal_noise = True if args.normal_noise else False
    binarize = True if args.binarize else False

    # for c0 in [6]:
    #     for c1 in [8]:
    #         print('c0 (%d) vs c1 (%d)' % (c0, c1))
    #         train = np.concatenate([train_[train_label_ == c0], train_[train_label_ == c1]], axis=0)
    #         c0_shape = (train_label_ == c0).sum()
    #         train_label = np.zeros((train.shape[0],), dtype=np.int8)
    #         train_label[c0_shape:] = 1
    #
    #         test = np.concatenate([test_[test_label_ == c0], test_[test_label_ == c1]], axis=0)
    #         c0_shape = (test_label_ == c0).sum()
    #         test_label = np.zeros((test.shape[0],), dtype=np.int8)
    #         test_label[c0_shape:] = 1





            # if binarize:
            #     def bi(x, p):
            #         return np.sign(2 * x - 1) * 0.5 * np.power(np.abs(2 * x - 1), 2 / p) + 0.5
            #
            #
            #     train = bi(train, args.eps)
            #     test = bi(test, args.eps)
            #
            # print('training data size: ')
            # print(train.shape)
            # print('testing data size: ')
            # print(test.shape)

    if normal_noise:
        noise = np.random.normal(0, 1, size=train.shape)
        noisy = np.clip((train + noise * args.epsilon), 0, 1)
        train = np.concatenate([train, noisy], axis=0)
        train_label = np.concatenate([train_label] * 2, axis=0)
    mlps = []
    for i in range(args.round):
        node_list = tuple([args.hidden_nodes] * args.h_times)
        mlps.append(MLPClassifier(hidden_layer_sizes=node_list, activation='logistic', solver='sgd',
                            alpha=0.0001,
                            batch_size='auto',
                            learning_rate='constant', learning_rate_init=args.lr, power_t=0.5, max_iter=args.iters,
                            shuffle=True,
                            random_state=i, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                            nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                            beta_2=0.999,
                            epsilon=1e-08, n_iter_no_change=10,))
    models = []
    for i in range(args.round):
        models.append((str(i), mlps[i]))
    a = time.time()
    scd = VotingClassifier(estimators=models, voting='soft')
    scd.fit(train, train_label)
    print('Cost: %.3f seconds' % (time.time() - a))

    print('Cost: %.3f seconds' % (time.time() - a))

    print('Best Train Accuracy: ', accuracy_score(y_true=train_label, y_pred=scd.predict(train)))
    print('Balanced Train Accuracy: ', balanced_accuracy_score(y_true=train_label, y_pred=scd.predict(train)))
    print('Best one Accuracy: ', accuracy_score(y_true=test_label, y_pred=scd.predict(test)))
    print('Balanced Accuracy: ', balanced_accuracy_score(y_true=test_label, y_pred=scd.predict(test)))

    if save:
        save_path = 'checkpoints'
        save_checkpoint(scd, save_path, args.target, et, vc)
    # del scd