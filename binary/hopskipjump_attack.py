import sys
sys.path.append('..')
import pickle
import os
from tools import args, save_checkpoint, print_title, load_data
import numpy as np
from art.attacks.evasion import HopSkipJump
from art.classifiers import BlackBoxClassifier
import time

class modelWrapper():
    def __init__(self, model):
        self.model = model

    def predict_one_hot(self, x_test, **kwargs):
        pred_y = self.model.predict(x_test)
        pred_one_hot = np.eye(args.n_classes)[pred_y.astype(int)]

        return pred_one_hot






if __name__ == '__main__':

    # load data
    train, test, train_label, test_label = load_data(args.dataset, args.n_classes)

    if 'lenet' in args.target:
        train = train.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2)).astype(np.float32)
        test = test.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2)).astype(np.float32)

    if 'approx' in args.target:
        scd = BNN(['checkpoints/%s_%d.h5' % (args.target, i) for i in range(100)])
    elif 'simclr' in args.target:
        scd = SimModel(sim_path='checkpoints/128_0.5_200_512_500_model.pth',
                       scd_path='checkpoints/%s' % args.target)

    elif 'resnet' in args.target:
            scd = CNNVote('checkpoints/%s' % args.target, 10)
    else:
        with open('checkpoints/%s' % args.target, 'rb') as f:
            scd = pickle.load(f)

    input_shape = train.shape[1:]
    yp = scd.predict(test)

    correct_index = np.nonzero((yp == test_label).astype(np.int8))[0]

    predictWrapper = modelWrapper(scd)

    min_pixel_value = train.min()
    max_pixel_value = train.max()
    print('min_pixel_value ', min_pixel_value)
    print('max_pixel_value ', max_pixel_value)


    # Create classifier
    classifier = BlackBoxClassifier(predict=predictWrapper.predict_one_hot,
                                    input_shape=input_shape,
                                    nb_classes=args.n_classes,
                                    clip_values=(min_pixel_value, max_pixel_value))

    print('----- generate adv data by HopSkipJump attack -----')
    # Generate adversarial test examples


    attacker = HopSkipJump(classifier=classifier, targeted=False, norm=2, max_iter=40, max_eval=10000, init_eval=100,
                           init_size=100)
    # attacker = HopSkipJump(classifier=classifier, targeted=False, norm=2, max_iter=2, max_eval=10000, init_eval=100, init_size=100)

    # Input data shape should be 2D
    datapoint = test[correct_index[:1]]

    s = time.time()
    adv_data = attacker.generate(x=datapoint)

    # distortion(datapoint, adv_data)
    print('Generate test adv cost time: ', time.time() - s)

    # return adv_data