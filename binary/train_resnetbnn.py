import tensorflow as tf
import larq as lq
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from tools import args, load_data, ModelArgs, BalancedBatchSampler
from core.bnn import BNN
import pickle
# from larq_zoo.literature.resnet_e import BinaryResNetE18Factory
from core.resnet_e import BinaryResNetE18Factory

np.random.seed(args.seed)

train_data, test_data, train_label, test_label = load_data(args.dataset, args.n_classes)
# train_data = train_data.astype(np.float32).reshape((-1, 3, 224, 224)).transpose((0, 2, 3, 1))
# test_data = test_data.astype(np.float32).reshape((-1, 3, 224, 224)).transpose((0, 2, 3, 1))
train_data = train_data.astype(np.float32).reshape((-1, 32, 32, 3))
test_data = test_data.astype(np.float32).reshape((-1, 32, 32, 3))

if args.normal_noise:
    noise = np.random.normal(0, 1, size=train_data.shape)
    noisy = np.clip((train_data + noise * args.epsilon), 0, 1)
    train_data = np.concatenate([train_data, noisy], axis=0)
    train_label = np.concatenate([train_label] * 2, axis=0)

train_labels = tf.keras.utils.to_categorical(train_label, args.n_classes)
test_labels = tf.keras.utils.to_categorical(test_label, args.n_classes)

kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="approx_sign",
              kernel_constraint="weight_clip",
              use_bias=False)

for i in range(args.round):
    model = BinaryResNetE18Factory(input_shape=(32, 32, 3), weights=None, num_classes=2)
    model = model.build()

    model.compile(
        tf.keras.optimizers.Adam(lr=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    trained_model = model.fit(
        train_data,
        train_labels,
        batch_size=64,
        epochs=100,
        validation_data=(test_data, test_labels),
        shuffle=True
    )
    model.save('checkpoints/%s_%d.h5' % (args.target, i))
    del model, trained_model
