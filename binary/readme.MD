Binary Classification and Black-box Attack
=

Introduction
-

**Training scripts parameters for different benchmark descripted in `job` directory**

1. Preparing data

    Experiments include three benchmarks data set. MNIST, CIFAR10, IMAGENET.
    
    MNIST data set is downloaded through `torchvision`. You can go to `tools`, 
    run `get_data('mnist')` in `dataset.py`. It will download data into `data`
    directory.
    
    CIFAR10 data set is downloaded through `torchvision`. You can go to `tools`, 
    run `get_data('cifar10')` in `dataset.py`. It will download data into `data`
    directory.
    
    IMAGENET data set should be prepared by yourself. Because we only use 10 classes
    from the full dataset and split them into training and testing manually, we 
    provide file name list `train_mc` and `val_mc` in `data` to help you splitting them. Please modify the code `get_data` in `tools`
    to access your imagenet data path.
    
2. Train SCD models

    Train a scd model, includes scd01mlp, scd01mlp-bnn, 
    scd01-cross-entropy-mlp, scd01-cross-entropy-mlp-bnn. All variations are trained through 
    `train_scd4.py`.
    
    Example
    
        python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
        --b-ratio 0.2 --updated-features 128 --round 100 --interval 20 --n-jobs 2 --num-gpus 2  --save --n_classes 2 --iters 1 \
        --target cifar10_scdcemlp_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset cifar10 --version ce --seed 0 --width 10000 \
        --metrics balanced --init normal  --updated-nodes 1
    
    Args explanation
    
    -   `nrows` The data sampling ratio in each iteration. `0.75` means 75% data from **each class** will be randomly picked
    in each iteration.
    
    -   `nfeatures` The feature sampling ratio in each vote. `1` means all features will be used in each vote.
    `0.5` means 50% features randomly picked will be used in each vote.
    
    -   `w-inc1` The step size for the first hidden layer.
    
    -   `w-inc2` The step size for the second layer (the last layer in our network).
    
    -   `hidden-nodes` The number of nodes in the hidden layer.
    
    -   `num-iters`  The number of iterations.

    -   `b-ratio` Balanced ratio of iterations during the training. `0.2` means 
    balanced accuracy evaluation for the loss will be used in the first 20% iterations. 
    In binary classification, ignore this parameters.
    
    -   `updated-features` The number of features will be considered to update in each iteration. 
    `128` means scd will go over randomly 128 features' updating with the same step size
     and keep the best one among these 128 features.

    -   `round` The number of votes.
    
    -   `interval` How many neighbors will be consider in local bias search.
    
    -   `n-jobs` The number of threads, set it equals to `num-gpus` is fine.
    
    -   `num-gpus` The number of GPUs, the votes will be balanced assigned to each GPU. 
    More GPUs works only for multiple votes.

    -   `save` Whether to save the model.
    
    -   `n_classes` The number of classes, a parameter for loading data.

    -   `iters` The maximum local iterations in scd search. Bigger iterations means deeper search.
    
    -   `target`  Save model named as `target`.
    
    -   `dataset` Data set name, a parameter for loading data.
    
    -   `version` 
        **mlp** is scd optimized 01 loss two layers MLP.
        **ce** is scd optimized cross-entropy loss, sign activated two layers MLP.
        **01bnn** is scd optimized 01 loss two layers binarized weights (+1 and -1 only) MLP.
        **cebnn** is scd optimized cross-entropy loss, sign activated two layers MLP.
    
    -   `seed`  Global random seed.

    -   `width` A parameter to save GPU memory usage. If program give you a **"out of GPU memory"**
error, reduce it.    
    -   `metrics`   Ignore it in binary classification

    -   `init` Distribution of initialized weights. `normal` means Normal distribution, `uniform` 
    means uniform distribution. 
    
    -   `updated-nodes`  How many nodes in the hidden layer will be updated in each iteration. `1` means
    randomly pick one node to update in each iteration.
    
3. Train MLP models

    Single run and majority vote are in `train_mlp_major_.py`
    
    Example
    
        python train_mlp_major_.py --dataset cifar10 --n_classes 2 --hidden-nodes 20 --lr 0.01 --iters 1000 \
        --seed 2018 --save --target cifar10_mlp_100.pkl --round 100
        
4. Train BNN (Binarized Neural Network) models

    [Official Document](https://docs.larq.dev/larq/guides/bnn-optimization/)
    
    Training script is `train_binarynn.py`
    
    Example
    
        python train_binarynn.py --dataset cifar10 --n_classes 2 --hidden-nodes 20 --seed 2018 \
        --target cifar10_mlpbnn_approx --round 8 --batch-size 64 --lr 0.001 --epoch 100

5. Train ResNet and LeNet

    Training script is `train_lenet_.py`
    
    Please modify the code to make sure it use the correct structure. Because **ResNet18** or **ResNet50**
    are very big, please reduce the number of votes. Set  `--round` to less than 10 is safe.
    
    Example
    
        python train_lenet_.py --dataset cifar10 --n_classes 2 --round 100 --save \
        --target cifar10_lenet_100.pkl --seed 2018


6. HopSkipJump attack

    [IBM 360 ART library](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
    
    Our HopSkipJum attacking script is `hopskipjump_attack.py`
    
    Example
    
        python hopskipjump_attackpy --dataset cifar10 --n_classes 2 \
        --target cifar10_scd01mlp_100_br02_h20_nr075_ni1000_i1_0.pkl
        
    Please modify the script to satisfy your requirement. The blackbox-model wrapper need to set
    **the number of classes**. `predictor` need output a one-hot prediction vector.