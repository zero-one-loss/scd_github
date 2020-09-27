import tensorflow as tf
import larq as lq
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from tools import args, load_data, ModelArgs, BalancedBatchSampler
from core.bnn import BNN
import pickle
import time
np.random.seed(args.seed)

train_data, test_data, train_label, test_label = load_data(args.dataset, args.n_classes)
train_data = train_data.astype(np.float32)
test_data = test_data.astype(np.float32)

if args.normal_noise:
    noise = np.random.normal(0, 1, size=train_data.shape)
    noisy = np.clip((train_data + noise * args.epsilon), 0, 1)
    train_data = np.concatenate([train_data, noisy], axis=0)
    train_label = np.concatenate([train_label] * 2, axis=0)

train_labels = tf.keras.utils.to_categorical(train_label, args.n_classes)
test_labels = tf.keras.utils.to_categorical(test_label, args.n_classes)


kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip",
              use_bias=False)

for i in range(args.round):
    model = tf.keras.models.Sequential([
        # In the first layer we only quantize the weights and not the input

        lq.layers.QuantDense(args.hidden_nodes, kernel_quantizer="ste_sign",
                  kernel_constraint="weight_clip", ),
        tf.keras.layers.BatchNormalization(momentum=0.9, scale=False),
        # lq.layers.QuantDense(400, **kwargs),
        # tf.keras.layers.BatchNormalization(momentum=0.9, scale=False),
        # lq.layers.QuantDense(400, **kwargs),
        # tf.keras.layers.BatchNormalization(momentum=0.9, scale=False),
        # lq.layers.QuantDense(400, **kwargs),
        # tf.keras.layers.BatchNormalization(momentum=0.9, scale=False),
        lq.layers.QuantDense(args.n_classes, **kwargs),

        tf.keras.layers.Activation("softmax")
    ])

    model.compile(
        tf.keras.optimizers.Adam(lr=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    start_time = time.time()
    trained_model = model.fit(
        train_data,
        train_labels,
        batch_size=args.batch_size,
        epochs=args.epoch,
        validation_data=(test_data, test_labels),
        shuffle=True
    )
    print("training cost %.1f seconds" % (time.time() - start_time))
    save = True if args.save else False
    if save:
        model.save('checkpoints/%s_%d.h5' % (args.target, i))
    del model, trained_model
