Parameters
=

SCD models
-

| Dataset\Models | nrows | 1 | w-inc1 |w-inc2 | hidden-nodes | num-iters | updated-features | round | interval | version | init | 100vote Accuracy | Single vote Runtime (Second)|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| CIFAR10 |  |  |  | |  |  |  |  |  |  |  |  | |
| SCD01MLP| 0.75 | 1 | 0.17 |0.2 | 20 | 1000 | 128 | 100 | 20 | mlp | normal | 0.878 | 64 |
| SCD-CE-MLP| 0.75 | 1 | 0.17 |0.2 | 20 | 1000 | 128 | 100 | 20 | ce | normal | 0.882 | 56 |
| SCD-01MLP-bnn| 0.75 | 1 | 0.17 |0.2 | 20 | 10000 | 128 | 100 | 20 | 01bnn | normal | 0.861 | 467 |
|SCD-CE-MLP-BNN| 0.75 | 1 | 0.17 |0.2 | 20 | 10000 | 128 | 100 | 20 | cebnn | normal | 0.8675 | 422 |
| ImageNet |  |  |  | |  |  |  |  |  |  |  |  | |
| SCD01MLP| 0.75 | 1 | 0.17 |0.2 | 20 | 1000 | 256 | 100 | 20 | mlp | normal | 0.715 | 77 |
| SCD-CE-MLP| 0.75 | 1 | 0.17 |0.2 | 20 | 1000 | 256 | 100 | 20 | ce | normal | 0.7315 | 54 |
| SCD-01MLP-bnn| 0.75 | 1 | 0.17 |0.2 | 20 | 10000 | 256 | 100 | 20 | 01bnn | normal | 0.697 | 344 |
|SCD-CE-MLP-BNN| 0.75 | 1 | 0.17 |0.2 | 20 | 10000 | 256 | 100 | 20 | cebnn | normal | 0.706 | 337 |
| GTSRB |  |  |  | |  |  |  |  |  |  |  |  | |
| SCD01MLP| 0.25 | 1 | 0.1 |0.2 | 20 | 1000 | 256 | 100 | 20 | mlp | normal | 0.974 | 22 |
| SCD-CE-MLP| 0.25 | 1 | 0.1 |0.2 | 20 | 1000 | 256 | 100 | 20 | ce | normal | 0.975 | 22 |
| SCD-01MLP-bnn| 0.25 | 1 | 0.17 |0.2 | 20 | 10000 | 256 | 100 | 20 | 01bnn | normal | 0.976 | 85 |
|SCD-CE-MLP-BNN| 0.25 | 1 | 0.17 |0.2 | 20 | 10000 | 256 | 100 | 20 | cebnn | normal | 0.98 | 91 |
| Celeba |  |  |  | |  |  |  |  |  |  |  |  | |
| SCD01MLP| 0.75 | 1 | 0.17 |0.2 | 20 | 1000 | 256 | 100 | 20 | mlp | normal | 0.79 | 20 |
| SCD-CE-MLP| 0.75 | 1 | 0.17 |0.2 | 20 | 1000 | 256 | 100 | 20 | ce | normal | 0.813 | 18 |
| SCD-01MLP-bnn| 0.75 | 1 | 0.17 |0.2 | 20 | 10000 | 256 | 100 | 20 | 01bnn | normal | 0.741 | 105 |
|SCD-CE-MLP-BNN| 0.75 | 1 | 0.17 |0.2 | 20 | 10000 | 256 | 100 | 20 | cebnn | normal | 0.723 | 111 |