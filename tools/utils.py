import time
import os
import pickle
import numpy as np
import sys

def save_checkpoint(obj, save_path, file_name, et, vc):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    suffix = '%s'*6 % et[:6]
    new_name, extend = os.path.splitext(file_name)
    new_name = "%s_%s%s" % (new_name, suffix, extend)
    full_name = os.path.join(os.getcwd(), os.path.join(save_path, file_name))
    with open(full_name, 'wb') as f:
        pickle.dump(obj, f)
    print('Save %s successfully, verification code: %s' % (full_name, vc))


def print_title(vc_len=4):
    vc_table = [chr(i) for i in range(97, 123)]
    vc = ''.join(np.random.choice(vc_table, vc_len))
    print(' ')
    et = time.localtime()
    print('Experiment time: ', time.strftime("%Y-%m-%d %H:%M:%S", et))
    print('Verification code: ', vc)
    print('Args:')
    print(sys.argv)

    return et, vc


import torch

is_torchvision_installed = True
try:
    import torchvision
except:
    is_torchvision_installed = False
import torch.utils.data
import random


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1] * len(self.keys)

    def _get_label(self, dataset, idx, labels=None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if is_torchvision_installed and dataset_type is torchvision.datasets.MNIST:
                return dataset.train_labels[idx].item()
            elif is_torchvision_installed and dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max * len(self.keys)