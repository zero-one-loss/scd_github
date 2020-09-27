import numpy as np
import pickle
import sys
from torchvision.models.resnet import resnet50, resnet18
sys.path.append('..')
from tools import args, save_checkpoint, print_title, load_data
import torch.nn as nn

if args.version == 'linear':
    from core.scd01_binary import SCD
    # basic scd 01 loss multi-class one vs all linear classifier

elif args.version == 'mlp':
    from core.scd01mlp_binary import SCD
    # scd mlp 01 multi-class one vs all

elif args.version == 'v1':
    from core.v1 import SCD
    # scd mlp 01 multi-class one vs all

import time
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from sklearn.model_selection import train_test_split
from skorch.helper import predefined_split
from skorch.dataset import Dataset
from core.cifar10_net import CNN, LeNet_gtsrb, SimpleNet_gtsrb, LeNet_celeba, SimpleNet_celeba, LeNet_cifar, SimpleNet_cifar

from core.cnn_ensemble import CNN
# from core.resnet import ResNet18


if __name__ == '__main__':
    # Set Random Seed

    # print information
    et, vc = print_title()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    save = True if args.save else False
    random_patch = True if args.random_patch else False
    normal_noise = True if args.normal_noise else False
    binarize = True if args.binarize else False

    train, test, train_label, test_label = load_data(args.dataset, args.n_classes)

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

    if random_patch:
        if args.dataset == 'mnist':
            shape = (28, 28)
        elif args.dataset == 'cifar10':
            shape = (32, 32, 3)
        elif args.dataset == 'gtsrb_binary':
            shape = (3, 48, 48)
        elif args.dataset == 'celeba':
            shape = (3, 96, 96)
        elif args.dataset == 'imagenet':
            shape = (3, 224, 224)
    else:
        shape = None

    if normal_noise:
        noise = np.random.normal(0, 1, size=train.shape)
        noisy = np.clip((train + noise * args.epsilon), 0, 1)
        train = np.concatenate([train, noisy], axis=0)
        train_label = np.concatenate([train_label] * 2, axis=0)

    if binarize:
        def bi(x, p):
            return np.sign(2 * x - 1) * 0.5 * np.power(np.abs(2 * x - 1), 2 / p) + 0.5


        train = bi(train, args.eps)
        test = bi(test, args.eps)

    train = train.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2)).astype(np.float32)
    test = test.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2)).astype(np.float32)
    # train = train.reshape((-1, 3, 48, 48)).astype(np.float32)
    # test = test.reshape((-1, 3, 48, 48)).astype(np.float32)
    # train = train.reshape((-1, 3, 96, 96)).astype(np.float32)
    # test = test.reshape((-1, 3, 96, 96)).astype(np.float32)
    # train = train.reshape((-1, 3, 224, 224)).astype(np.float32)
    # test = test.reshape((-1, 3, 224, 224)).astype(np.float32)
    train_label = train_label.astype(np.int64)
    test_label = test_label.astype(np.int64)
    print('training data size: ')
    print(train.shape)
    print('testing data size: ')
    print(test.shape)

    valid_ds = Dataset(test, test_label)

    # scd = NeuralNetClassifier(
    #     SimpleNet,
    #     classes=2,
    #     max_epochs=30,
    #     lr=0.001,
    #     criterion=torch.nn.CrossEntropyLoss,
    #     # Shuffle training data on each epoch
    #     iterator_train__shuffle=True,
    #     train_split=predefined_split(valid_ds),
    #     batch_size=256,
    #     optimizer=torch.optim.Adam,
    #     device='cuda',
    #     verbose=1,
    #     warm_start=False,
    # )
    # scd = CNN(args.round, SimpleNet_gtsrb)
    # scd = CNN(args.round, LeNet_gtsrb)
    # scd = CNN(args.round, SimpleNet_celeba)
    # scd = CNN(args.round, LeNet_celeba)
    # scd = CNN(args.round, SimpleNet_cifar)
    scd = CNN(args.round, LeNet_cifar)
    # scd = CNN(args.round, ResNet18)
    # models = []
    # for i in range(args.round):
    #     model = resnet50()
    #     model._modules['conv1'] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    #     model._modules['fc'] = nn.Linear(2048, args.n_classes, bias=True)
    #
    #     models.append(model)
    # scd = CNN(args.round, models)
    # scd = CNN(args.round, resnet50)
    # scd = CNN(args.round, resnet18)
    a = time.time()
    scd.fit(train, train_label)

    print('Cost: %.3f seconds' % (time.time() - a))

    print('Best Train Accuracy: ', accuracy_score(y_true=train_label, y_pred=scd.predict(train)))
    # print('Vote Train Accuracy: ', accuracy_score(y_true=train_label, y_pred=scd.predict(train)))
    print('Best one Accuracy: ', accuracy_score(y_true=test_label, y_pred=scd.predict(test)))
    # print('vote  Accuracy: ', accuracy_score(y_true=test_label, y_pred=scd.predict(test)))

    if save:
        save_path = 'checkpoints'
        save_checkpoint(scd, save_path, args.target, et, vc)
    # del scd