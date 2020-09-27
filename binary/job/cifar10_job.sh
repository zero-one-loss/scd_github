#!/bin/bash -l

#SBATCH -p datasci


#SBATCH --job-name=cifar10_job
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

cd ..


python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
--b-ratio 0.2 --updated-features 128 --round 100 --interval 20 --n-jobs 2 --num-gpus 2  --save --n_classes 2 --iters 1 \
--target cifar10_scdcemlp_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset cifar10 --version ce --seed 0 --width 10000 \
--metrics balanced --init normal  --updated-nodes 1



#
python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 10000 \
--b-ratio 0.2 --updated-features 128 --round 100 --interval 20 --n-jobs 2 --num-gpus 2  --save --n_classes 2 \
--iters 1 --target cifar10_scd01mlpbnn_100_br02_h20_nr075_ni10000_i1_0.pkl --dataset cifar10 --version 01bnn --seed 0 \
--width 10000 --metrics balanced --init normal  --updated-nodes 1
#

#
python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 10000 \
--b-ratio 0.2 --updated-features 128 --round 100 --interval 20 --n-jobs 2 --num-gpus 2  --save --n_classes 2 \
--iters 1 --target cifar10_scdcemlpbnn_100_br02_h20_nr075_ni10000_i1_0.pkl --dataset cifar10 --version cebnn --seed 0 \
--width 10000 --metrics balanced --init normal  --updated-nodes 1

#python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 100 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 2 --iters 1 \
#--target imdb_scd01mlp_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset imdb --version mlp --seed 0 --width 25000 \
#--metrics balanced --init normal  --updated-nodes 1
#
#python train_mlp_major_.py --dataset imdb --n_classes 2 --hidden-nodes 20 --lr 0.1 --iters 1000 --seed 2018 --save --target imdb_mlp_20.pkl --round 8
#python train_mlp_major_.py --dataset imdb --n_classes 2 --hidden-nodes 200 --lr 0.1 --iters 1000 --seed 2018 --save --target imdb_mlp_200.pkl --round 8

#python train_mlp_major_.py --dataset imdb --n_classes 2 --hidden-nodes 20 --lr 0.1 --iters 1000 --seed 2018 --save --target imdb_mlp_10.pkl --round 8

#python train_binarynn.py --dataset imdb --n_classes 2 --hidden-nodes 20 --seed 2018 \
#--target imdb_mlpbnn_approx --round 8

#python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 \
#--updated-features 128 --round 4 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 2 --iters 1 \
#--target cifar10_scdcemlp_100_br02_h20_nr075_ni1000_i1_0.pkl --dataset cifar10 --version ce --seed 0 --width \
#10000 \
#--metrics balanced --init normal  --updated-nodes 1
##
#python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 16 --num-iters 800 \
#--b-ratio 0.2 \
#--updated-features 128 --round 100 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 2 --iters 1 \
#--target cifar10_scd01mlp_100_br02_h16_nr075_ni1000_i1_0.pkl --dataset cifar10 --version mlp --seed 0 --width \
#10000 \
#--metrics balanced --init normal  --updated-nodes 1
#
#python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 32 --num-iters 1600 \
#--b-ratio 0.2 \
#--updated-features 128 --round 100 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 2 --iters 1 \
#--target cifar10_scd01mlp_100_br02_h32_nr075_ni1000_i1_0.pkl --dataset cifar10 --version mlp --seed 0 --width \
#10000 \
#--metrics balanced --init normal  --updated-nodes 1
#
#python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 64 --num-iters 3200 \
#--b-ratio 0.2 \
#--updated-features 128 --round 100 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 2 --iters 1 \
#--target cifar10_scd01mlp_100_br02_h64_nr075_ni1000_i1_0.pkl --dataset cifar10 --version mlp --seed 0 --width \
#10000 \
#--metrics balanced --init normal  --updated-nodes 1
#
#python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 \
#--updated-features 128 --round 32 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 2 --iters 1 \
#--target cifar10_simclr_scd01mlp_32_br02_h20_nr075_ni1000_i1_0.pkl --dataset cifar10_simclr --version mlp --seed 0 \
#--width \
#10000 \
#--metrics balanced --init normal  --updated-nodes 1 --verbose

#python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 256 --num-iters 12800 \
#--b-ratio 0.2 \
#--updated-features 128 --round 100 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 2 --iters 1 \
#--target cifar10_scd01mlp_100_br02_h256_nr075_ni1000_i1_0.pkl --dataset cifar10 --version mlp --seed 0 --width \
#10000 \
#--metrics balanced --init normal  --updated-nodes 1
#
#python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 512 --num-iters 25600 \
#--b-ratio 0.2 \
#--updated-features 128 --round 100 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 2 --iters 1 \
#--target cifar10_scd01mlp_100_br02_h512_nr075_ni1000_i1_0.pkl --dataset cifar10 --version mlp --seed 0 --width \
#10000 \
#--metrics balanced --init normal  --updated-nodes 1

#python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 \
#--updated-features 128 --round 128 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 2 --iters 1 \
#--target cifar10_scd01mlp_128_br02_h20_nr075_ni1_i1_0.pkl --dataset cifar10 --version mlp --seed 0 --width \
#10000 \
#--metrics balanced --init normal  --updated-nodes 1
#
#python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 \
#--updated-features 128 --round 256 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 2 --iters 1 \
#--target cifar10_scd01mlp_256_br02_h20_nr075_ni1_i1_0.pkl --dataset cifar10 --version mlp --seed 0 --width \
#10000 \
#--metrics balanced --init normal  --updated-nodes 1

#python train_scd4.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 1000 \
#--b-ratio 0.2 \
#--updated-features 128 --round 100 --interval 20 --n-jobs 4 --num-gpus 4  --save --n_classes 2 --iters 1 \
#--target cifar10_scd01_100_br02_nr075_ni1000_i1_0.pkl --dataset cifar10 --version linear --seed 0 --width \
#10000 \
#--metrics balanced --init normal  --updated-nodes 1


#python train_lenet_.py --dataset cifar10_binary --seed 2018 --n_classes 2 \
#--save --target cifar10_binary_lenet_100.pkl --round 100
#
#python train_lenet_.py --dataset cifar10_binary --n_classes 2 --round 100 --seed 2018 \
#--target cifar10_binary_lenet_100_ep1.pkl --save --normal-noise --epsilon 0.1
#
#python train_lenet_.py --dataset cifar10_binary --n_classes 2 --round 100 --seed 2018 \
#--target cifar10_binary_lenet_100_ep2.pkl --save --normal-noise --epsilon 0.2

#python train_lenet_.py --dataset cifar10_binary --seed 2018 --n_classes 2 \
#--save --target cifar10_binary_simplenet_100.pkl --round 100
#
#python train_lenet_.py --dataset cifar10_binary --n_classes 2 --round 100 --seed 2018 \
#--target cifar10_binary_simplenet_100_ep1.pkl --save --normal-noise --epsilon 0.1
#
#python train_lenet_.py --dataset cifar10_binary --n_classes 2 --round 100 --seed 2018 \
#--target cifar10_binary_simplenet_100_ep2.pkl --save --normal-noise --epsilon 0.2

#gpu=0
#version=mlp1
#seed=2023
#
#python train_mlp_ml_4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 \
#--num-iters 1000 --b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 10 \
#--iters 1 --updated-nodes 1 --target cifar10_1_nr075_${version}_h20_sign_01loss_1000_${seed}.pkl --dataset cifar10  \
#--seed $seed  --version $version --act sign


#python train_binarynn.py --dataset cifar10 --n_classes 2 --hidden-nodes 20 --seed 2018 \
#--target cifar10_mlpbnn_approx --round 256

#python train_binarynn.py --dataset cifar10 --n_classes 2 --hidden-nodes 20 --seed 2018 \
#--target cifar10_mlpbnn_ste --round 100

#python train_binarynn.py --dataset cifar10 --n_classes 2 --hidden-nodes 20 --seed 2018 \
#--target cifar10_mlpbnn_swish --round 100

#
#python train_binarynn.py --dataset cifar10 --n_classes 2 --hidden-nodes 20 --seed 2018 \
#--target cifar10_mlpbnn_approx_ep1 --round 100 --normal-noise --epsilon 0.1
#
#python train_binarynn.py --dataset cifar10 --n_classes 2 --hidden-nodes 20 --seed 2018 \
#--target cifar10_mlpbnn_approx_ep004 --round 100 --normal-noise --epsilon 0.004
#
#python train_binarynn.py --dataset cifar10 --n_classes 2 --hidden-nodes 20 --seed 2018 \
#--target cifar10_mlpbnn_approx_ep2 --round 100 --normal-noise --epsilon 0.2

#version=mlp1
#seed=1

#python train_mlp_ml_4.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 160 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
#--target cifar10_1_nr075_${version}_sign_01loss_1000_${seed}_ep1.pkl --dataset cifar10  --seed $seed  --version $version \
#--act sign

#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
#--target cifar10_1_nr075_${version}_sign_01loss_1000_${seed}_ep2.pkl --dataset cifar10  --seed $seed  --version $version \
#--act sign --normal-noise --epsilon 0.2
#
#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
#--target cifar10_1_nr075_${version}_sign_01loss_1000_${seed}_ep3.pkl --dataset cifar10  --seed $seed  --version $version \
#--act sign --normal-noise --epsilon 0.3
#
#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
#--target cifar10_1_nr075_${version}_sign_01loss_1000_${seed}_w10_h1.pkl --dataset cifar10  --seed $seed  --version $version \
#--act sign --width 10

##
#python train_mlp_ml_3.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
#--target cifar10_1_nr075_${version}_sign_01loss_2000_5006.pkl --dataset cifar10  --seed 5006  --version $version --act sign
#
#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
#--target cifar10_1_nr075_${version}_sign_01loss_2000_2.pkl --dataset cifar10  --seed 2  --version $version --act sign
#
#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
#--target cifar10_1_nr075_${version}_sign_01loss_2000_3.pkl --dataset cifar10  --seed 3  --version $version --act sign

#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.1 --w-inc2 0.1 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
#--target cifar10_1_nr075_${version}_sign_01loss_2000_0_w1_h1_0101.pkl --dataset cifar10  --seed 0  \
#--version $version --act sign --width 1
#
#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.05 --w-inc2 0.05 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
#--target cifar10_1_nr075_${version}_sign_01loss_2000_0_w1_h1_005005.pkl --dataset cifar10  --seed 0  \
#--version $version --act sign --width 1
#
#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.05 --w-inc2 0.5 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
#--target cifar10_1_nr075_${version}_sign_01loss_2000_0_w1_h1_00505.pkl --dataset cifar10  --seed 0  \
#--version $version --act sign --width 1
#
#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.1 --w-inc2 0.5 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
#--target cifar10_1_nr075_${version}_sign_01loss_2000_0_w1_h1_0105.pkl --dataset cifar10  --seed 0  \
#--version $version --act sign --width 1

#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
#--target cifar10_1_nr075_${version}_sign_01loss_2000_0_w10_h1.pkl --dataset cifar10  --seed 0  \
#--version $version --act sign --width 10
#
#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 120 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 20 \
#--target cifar10_1_nr075_${version}_sign_01loss_120_0_w1_h20.pkl --dataset cifar10  --seed 0  \
#--version $version --act sign --width 1


#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target gtsrb_binary_1_nr075_${version}_sign_01loss_2000_3.pkl --dataset gtsrb_binary  --seed 3  --version $version --act sign

#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_1_nr075_mlp1_sign_01loss_2000_4.pkl --dataset cifar10  --seed 4  --version mlp1 --act sign
#
#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_1_nr075_mlp1_sign_01loss_2000_5.pkl --dataset cifar10  --seed 5  --version mlp1 --act sign
#
#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_1_nr075_mlp1_sign_01loss_2000_6.pkl --dataset cifar10  --seed 6  --version mlp1 --act sign
##
#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_1_nr075_mlp1_sign_01loss_2000_7.pkl --dataset cifar10  --seed 7  --version mlp1 --act sign



#python train_scd_vote.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 4 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_4_nr075_mlp_sign_01loss_2000_i20_2019.pkl --dataset cifar10  --seed 2019  --version mlp --act sign \
#--n-jobs 4 --num-gpus 4
#
#python train_scd_vote.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 4 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_4_nr075_mlp2_sign_01loss_2000_i20_2019.pkl --dataset cifar10  --seed 2019  --version mlp2 --act sign \
#--n-jobs 4 --num-gpus 4
#
#python train_scd_vote.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 4 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_4_nr075_mlp3_sign_01loss_2000.pkl --dataset cifar10  --seed 2019  --version mlp3 --act sign \
#--n-jobs 4 --num-gpus 4

#python train_scd_vote.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 4 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_4_nr075_mlp4_sign_01loss_2000.pkl --dataset cifar10  --seed 2019  --version mlp4 --act sign \
#--n-jobs 4 --num-gpus 4
#
#python train_scd_vote.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 4 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_4_nr075_mlp5_sign_01loss_2000.pkl --dataset cifar10  --seed 2019  --version mlp5 --act sign \
#--n-jobs 4 --num-gpus 4
#
#python train_scd_vote.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 4 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_4_nr075_mlp6_sign_01loss_2000.pkl --dataset cifar10  --seed 2019  --version mlp6 --act sign \
#--n-jobs 4 --num-gpus 4

#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_1_nr075_mlp_sign_01loss_1000.pkl --dataset cifar10  --seed 2028  --version mlp --act sign

#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_1_nr075_mlp4_sign_01loss_2000.pkl --dataset cifar10  --seed 2028  --version mlp4 --act sign

#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_1_nr075_mlp3_sign_01loss_2000_i10.pkl --dataset cifar10  --seed 2028  --version mlp3 --act sign
#
#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_1_nr075_mlp3_sign_01loss_2000_i20.pkl --dataset cifar10  --seed 2028  --version mlp3 --act sign
#
#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_1_nr075_mlp3_sign_01loss_2000_i10_fp16.pkl --dataset cifar10  --seed 2028  --version mlp3 --act sign --fp16
#
#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_1_nr075_mlp3_sign_01loss_2000_i20_fp16.pkl --dataset cifar10  --seed 2028  --version mlp3 --act sign --fp16

#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_1_nr075_mlp5_sign_01loss_2000_i20.pkl --dataset cifar10  --seed 2028  --version mlp5 --act sign

#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_1_nr075_mlp6_sign_01loss_2000_i20.pkl --dataset cifar10  --seed 2028  --version mlp6 --act sign
#
#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_1_nr075_mlp2_sign_01loss_2000_i20.pkl --dataset cifar10  --seed 2028  --version mlp2 --act sign
#
#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_1_nr075_mlp_sign_01loss_2000_i20.pkl --dataset cifar10  --seed 2028  --version mlp --act sign

#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_1_nr075_mlp4_sign_01loss_2000_i20.pkl --dataset cifar10  --seed 2028  --version mlp4 --act sign


#
#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 10 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_1_nr075_mlp_sign_01l0ss_1000.pkl --dataset cifar10  --seed 2019  --version mlp --act sign > temp_log2

#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_1_nr075_mlp_sign_01l0ss_1000.pkl --dataset cifar10  --seed 2019  --version mlp --act sign
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_1_nr075_mlp_sign_01l0ss_1000 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2
#done

#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_2_nr075_mlp_sign_01l0ss_1000.pkl --dataset cifar10  --seed 2019  --version mlp --act sign
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_2_nr075_mlp_sign_01l0ss_1000 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2
#done

#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.1 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_3_nr075_mlp_sign_01l0ss_1000.pkl --dataset cifar10  --seed 2019  --version mlp --act sign
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_3_nr075_mlp_sign_01l0ss_1000 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2
#done


#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_4_nr075_mlp_sign_01l0ss_1000.pkl --dataset cifar10  --seed 2019  --version mlp2 --act sign
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_4_nr075_mlp_sign_01l0ss_1000 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2
#done

#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.1 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_5_nr075_mlp_sign_01l0ss_1000.pkl --dataset cifar10  --seed 2019  --version mlp2 --act sign
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_5_nr075_mlp_sign_01l0ss_1000 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2
#done

#python train_mlp_ml_.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.1 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_6_nr075_mlp_sign_01l0ss_1000.pkl --dataset cifar10  --seed 2019  --version mlp2 --act sign
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_6_nr075_mlp_sign_01l0ss_1000 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2
#done

#python train_mlp_ml.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_1_nr075_mlp2_relu_01loss.pkl --dataset cifar10  --seed 2019  --version mlp --act relu

#python train_mlp01.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 \
#--num-iters 2000 --b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 \
#--iters 1 --target cifar10_mlp01ac_01loss_1.pkl --dataset cifar10  --seed 2018

#python train_mlp01.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 \
#--num-iters 2000 --b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 \
#--iters 1 --target cifar10_mlp01ac_01loss_2019.pkl --dataset cifar10  --seed 2019
#
#python train_mlp_ml.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_temp.pkl --dataset cifar10  --seed 2019

#python train_scd_vote.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 8 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_8_nr075_mlp2_sign_01loss.pkl --dataset cifar10  --seed 2019 --n-jobs 8 --num-gpus 4 --version mlp --act sign
#
#python train_scd_vote.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 8 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_8_nr075_mlp2_relu_01loss.pkl --dataset cifar10  --seed 2019 --n-jobs 4 --num-gpus 4 --version mlp --act relu

#python train_scd_vote.py --nrows 0.25 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 8 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_8_nr025_mlp2_sign_01loss.pkl --dataset cifar10  --seed 2019 --n-jobs 8 --num-gpus 4 --version mlp --act sign
#
#python train_scd_vote.py --nrows 0.25 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 2000 \
#--b-ratio 0.2 --updated-features 128 --round 8 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_8_nr025_mlp2_relu_01loss.pkl --dataset cifar10  --seed 2019 --n-jobs 4 --num-gpus 4 --version mlp --act relu

#python train_mlp01.py --nrows 0.75 --nfeatures 1 --w-inc1 0.1 --w-inc2 0.1 --hidden-nodes 20 \
#--num-iters 2000 --b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 \
#--iters 1 --target cifar10_mlpreluac_01loss_1.pkl --dataset cifar10  --seed 2020

#python train_mlp01.py --nrows 0.75 --nfeatures 1 --w-inc1 0.1 --w-inc2 0.1 --hidden-nodes 20 \
#--num-iters 2000 --b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 \
#--iters 1 --target cifar10_mlpreluac_01loss_1.pkl --dataset cifar10  --seed 2019

#
#python train_mlp01.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 \
#--num-iters 1000 --b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 \
#--iters 1 --target cifar10_mlp01ac_01loss_1.pkl --dataset cifar10  --seed 2020 --c 0.2

#python train_mlphinge.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.1 --hidden-nodes 20 \
#--num-iters 2000 --b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 \
#--iters 1 --target cifar10_mlp01ac_1.pkl --dataset cifar10  --seed 2018 --c 0.1

#python train_mlphinge.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.1 --hidden-nodes 20 \
#--num-iters 2000 --b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 \
#--iters 1 --target cifar10_mlp01ac_5.pkl --dataset cifar10  --seed 2018 --c 0.5
#
#python train_mlphinge.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.1 --hidden-nodes 20 \
#--num-iters 2000 --b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 \
#--iters 1 --target cifar10_mlp01ac_10.pkl --dataset cifar10  --seed 2018 --c 1.0
#
#python train_mlphinge.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.1 --hidden-nodes 20 \
#--num-iters 2000 --b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 \
#--iters 1 --target cifar10_mlp01ac_15.pkl --dataset cifar10  --seed 2018 --c 1.5
#
#python train_mlphinge.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.1 --hidden-nodes 20 \
#--num-iters 2000 --b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 \
#--iters 1 --target cifar10_mlp01ac_20.pkl --dataset cifar10  --seed 2018 --c 2.0
#
#python train_mlphinge.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.1 --hidden-nodes 20 \
#--num-iters 2000 --b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 \
#--iters 1 --target cifar10_mlp01ac_25.pkl --dataset cifar10  --seed 2018 --c 2.5
#
#python train_mlphinge.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.1 --hidden-nodes 20 \
#--num-iters 2000 --b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 \
#--iters 1 --target cifar10_mlp01ac_30.pkl --dataset cifar10  --seed 2018 --c 3.0
#
#python train_mlphinge.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.1 --hidden-nodes 20 \
#--num-iters 2000 --b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 \
#--iters 1 --target cifar10_mlp01ac_35.pkl --dataset cifar10  --seed 2018 --c 3.5


#python train_mlphinge.py --nrows 0.75 --nfeatures 1 --w-inc1 0.1 --w-inc2 0.1 --hidden-nodes 20 \
#--num-iters 2000 --b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 \
#--iters 1 --target cifar10_mlpreluac_1.pkl --dataset cifar10  --seed 2018 --c 0.1

#python train_mlphinge.py --nrows 0.75 --nfeatures 1 --w-inc1 0.1 --w-inc2 0.1 --hidden-nodes 20 \
#--num-iters 2000 --b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 \
#--iters 1 --target cifar10_mlpreluac_5.pkl --dataset cifar10  --seed 2018 --c 0.5
#
#python train_mlphinge.py --nrows 0.75 --nfeatures 1 --w-inc1 0.1 --w-inc2 0.1 --hidden-nodes 20 \
#--num-iters 2000 --b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 \
#--iters 1 --target cifar10_mlpreluac_10.pkl --dataset cifar10  --seed 2018 --c 1.0
#
#python train_mlphinge.py --nrows 0.75 --nfeatures 1 --w-inc1 0.1 --w-inc2 0.1 --hidden-nodes 20 \
#--num-iters 2000 --b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 \
#--iters 1 --target cifar10_mlpreluac_15.pkl --dataset cifar10  --seed 2018 --c 1.5
#
#python train_mlphinge.py --nrows 0.75 --nfeatures 1 --w-inc1 0.1 --w-inc2 0.1 --hidden-nodes 20 \
#--num-iters 2000 --b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 \
#--iters 1 --target cifar10_mlpreluac_20.pkl --dataset cifar10  --seed 2018 --c 2.0
#
#python train_mlphinge.py --nrows 0.75 --nfeatures 1 --w-inc1 0.1 --w-inc2 0.1 --hidden-nodes 20 \
#--num-iters 2000 --b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 \
#--iters 1 --target cifar10_mlpreluac_25.pkl --dataset cifar10  --seed 2018 --c 2.5
#
#python train_mlphinge.py --nrows 0.75 --nfeatures 1 --w-inc1 0.1 --w-inc2 0.1 --hidden-nodes 20 \
#--num-iters 2000 --b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 \
#--iters 1 --target cifar10_mlpreluac_30.pkl --dataset cifar10  --seed 2018 --c 3.0
#
#python train_mlphinge.py --nrows 0.75 --nfeatures 1 --w-inc1 0.1 --w-inc2 0.1 --hidden-nodes 20 \
#--num-iters 2000 --b-ratio 0.2 --updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 \
#--iters 1 --target cifar10_mlpreluac_35.pkl --dataset cifar10  --seed 2018 --c 3.5


# scd01

#gpu=0

#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 1000 --b-ratio 0.2 \
#--updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_hinge_02.pkl --dataset cifar10  --seed 2018 --c 0.2
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_hinge_02 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2
#done




#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 1000 --b-ratio 0.2 \
#--updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_hinge_04.pkl --dataset cifar10  --seed 2018 --c 0.4
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_hinge_04 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2
#done
#
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 1000 --b-ratio 0.2 \
#--updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_hinge_06.pkl --dataset cifar10  --seed 2018 --c 0.6
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_hinge_06 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2
#done
#
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 1000 --b-ratio 0.2 \
#--updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_hinge_08.pkl --dataset cifar10  --seed 2018 --c 0.8
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_hinge_08 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2
#done
#
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 1000 --b-ratio 0.2 \
#--updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_hinge_10.pkl --dataset cifar10  --seed 2018 --c 1.0
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_hinge_10 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2
#done
#
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 1000 --b-ratio 0.2 \
#--updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_hinge_12.pkl --dataset cifar10  --seed 2018 --c 1.2
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_hinge_12 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2
#done
#
#
#
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 1000 --b-ratio 0.2 \
#--updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_hinge_14.pkl --dataset cifar10  --seed 2018 --c 1.4
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_hinge_14 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2
#done
#
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 1000 --b-ratio 0.2 \
#--updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_hinge_16.pkl --dataset cifar10  --seed 2018 --c 1.6
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_hinge_16 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2
#done
#
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 1000 --b-ratio 0.2 \
#--updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_hinge_18.pkl --dataset cifar10  --seed 2018 --c 1.8
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_hinge_18 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2
#done
#
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 1000 --b-ratio 0.2 \
#--updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_hinge_20.pkl --dataset cifar10  --seed 2018 --c 2.0
#
#for seed in 2019 2393 92382 232 12 58 954 758 451 2015687
#do
#python bb_attack.py --epsilon 0.0625 --Lambda 0.1 --gpu $gpu --epoch 20 --aug-epoch 20 --lr 0.0001 \
#--train-size 200 --target cifar10_hinge_20 --random-sign 1 --seed $seed --dataset cifar10 \
#--oracle-size 1024 --n_classes 2
#done


#python train_mlp_ml_2.py --nrows 0.75 --nfeatures 1 --w-inc1 0.05 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
#--target cifar10_1_nr075_${version}_sign_01loss_1000_2017_w1_h1.pkl --dataset cifar10  --seed 2017 \
#--version $version --act sign --width 1
#
#python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 1000 --b-ratio 0.2 \
#--updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_scd_2017.pkl --dataset cifar10  --seed 2017
#
#python train_scd2.py --nrows 0.75 --nfeatures 1 --w-inc 0.02 --num-iters 1000 --b-ratio 0.2 \
#--updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_scd_2017.pkl --dataset cifar10  --seed 2017
#
#python train_scd2.py --nrows 0.75 --nfeatures 1 --w-inc 0.02 --num-iters 100 --b-ratio 0.2 \
#--updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_nr075_scd_2017.pkl --dataset cifar10  --seed 2017 --source cifar10_nr075_scd_2017.pkl
#
#python train_scd2.py --nrows 0.75 --nfeatures 1 --w-inc 0.05 --num-iters 100 --b-ratio 0.2 \
#--updated-features 3072 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target cifar10_nr075_scd_2017_keep_05_100_3072.pkl --dataset cifar10  --seed 2017 \
#--source cifar10_nr075_scd_2017_keep_05_100_3072.pkl