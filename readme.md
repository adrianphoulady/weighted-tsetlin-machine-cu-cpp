# The Weighted Tsetlin Machine in CUDA C++
This is a CUDA C++ implementation of the [Weighted Tsetlin Machine](https://arxiv.org/abs/1911.12607).

## Contents

- [Usage](#usage)
- [Precontained Datasets](#precontained-datasets)
- [How to Fit a Machine](#how-to-fit-a-machine)
- [Fitting a Classic Tsetlin Machine](#fitting-a-classic-tsetlin-machine)
- [License](#license)

## Usage
The following shows a minimal example to fit a Weighted Tsetlin Machine on the `dddd` dataset .

```c++
#include "utils.cuh"

int main(int argc, char * const argv[]) {
    std::string experiment = "imdb";
    int clauses = 80000, threshold = 100, epochs = 35, verbose = 1;
    double p = .006, gamma = .002;
    bool shuffle = true;

    update(argc, argv, experiment, clauses, p, threshold, gamma, epochs, shuffle, verbose);
    fit(experiment, clauses, p, gamma, threshold, epochs, shuffle, verbose);

    return 0;
}
```
For the `dddd` dataset, you should make the files `dddd-train.data` and `dddd-test.data` available in the `data/` folder. The `.data` files consist of samples, each at one line, made up of the binary features followed by an integer label, all separated by white spaces.  
There are three different datasets available at [The Weighted Tsetlin Machine in C++](https://github.com/adrianphoulady/weighted-tsetlin-machine-cpp) for MNIST, IMDb, and Connect-4, which can be used here.

The function `fit`'s signature is
```c++
fit(experiment, clauses, p, gamma, threshold, epochs, shuffle, verbose)
```
where `shuffle` makes the training samples shuffle at each epoch, and every `verbose` trainings of the machine the test accuracy will be measured.   

Also, there is a helper function `update`, which updates the parameters to `fit` from command line provided options.
```c++
update(argc, argv, experiment, clauses, p, threshold, gamma, epochs, shuffle, verbose)
```

The options are as follows.
    
`-c clauses`: number of clauses  
`-p probability`: feedback probability  
`-t threshold`: threshold  
`-g gamma`: learning rate gamma  
`-e epochs`: number of epochs  
`-n seed`: new random at each run by inputting `0`, or otherwise, randoms with the initial seed value of `seed`   
`-s ifshuffle`: if shuffle the training set at each epoch  
`-v verbose`: test every `verbose` training  

## Precontained Datasets
There are datasets for IMDb and EMNIST contained in the repository. Other datasets from [the C++ implementation repository](https://github.com/adrianphoulady/weighted-tsetlin-machine-cpp) can be used, too.
The dataset `dddd' dataset can be prepared in the `data/` folder by running
```sh
$ python3 prepare-dddd-dataset.py
```
or, unzipping the `dddd.data.zip` file.

## How to Fit a Machine
To make a fitter, first `make` the project.

```sh
$ make
nvcc -std=c++11 -O3 -o experiment experiment.cu
```

Thereafter, for `dddd` dataset, run `experiment` with the desired parameters as follows.

```sh
$./experiment -x imdb -c 80000 -p .006 -g .002 -t 100 -e 35 
imdb - samples=25K, features=10000, classes=2 - clauses=80000, p=0.0060, gamma=0.00200, threshold=100 - tpb=1024, bpg=80
epoch 001 of training and testing - 0129s and 0068s -  89.22%  and  85.96%
epoch 002 of training and testing - 0120s and 0068s -  90.93%  and  86.90%
epoch 003 of training and testing - 0115s and 0068s -  93.20%  and  89.86%
.
.
.
epoch 029 of training and testing - 0101s and 0068s -  99.33%  and  90.72%
epoch 030 of training and testing - 0101s and 0068s -  99.30%  and  90.59%
epoch 031 of training and testing - 0101s and 0068s -  99.50%  and  90.83%
.
.
.
epoch 048 of training and testing - 0099s and 0068s -  99.95%  and  90.64%
epoch 049 of training and testing - 0099s and 0068s -  99.94%  and  90.75%
epoch 050 of training and testing - 0099s and 0068s -  99.95%  and  90.51%
total time: 02:41:28
```
Setting `-v` is useful when the testing takes very long to proceed.
```sh
$./experiment -x emnist -c 200000 -p .02 -g .008 -t 250 -e 500 -v 10 
emnist - samples=697K, features=784, classes=62 - clauses=200000, p=0.0200, gamma=0.00800, threshold=250 - tpb=1024, bpg=80
epoch 001 of training - 0919s
epoch 002 of training - 0895s
epoch 003 of training - 0890s
epoch 004 of training - 0886s
epoch 005 of training - 0884s
epoch 006 of training - 0882s
epoch 007 of training - 0880s
epoch 008 of training - 0879s
epoch 009 of training - 0878s
epoch 010 of training and testing - 0878s and 3851s -  84.89%  and  82.40%
epoch 011 of training - 0877s
.
.
.
epoch 159 of training - 0866s
epoch 160 of training and testing - 0866s and 3854s -  92.94%  and  84.27%
.
.
.
``` 
## Fitting a Classic Tsetlin Machine
For having a classic [Tsetlin Machine](https://arxiv.org/abs/1804.01508), just set the learning rate `gamma` to `0`.

```sh
$ ./experiment -x imdb -c 25000 -p .02 -g .0 -t 200 -e 3
imdb - samples=25K, features=10000, classes=2 - clauses=25000, p=0.0200, gamma=0.00000, threshold=200 - tpb=1024, bpg=80
epoch 001 of training and testing - 0042s and 0022s -  88.22%  and  86.20%
epoch 002 of training and testing - 0039s and 0022s -  89.58%  and  87.94%
epoch 003 of training and testing - 0039s and 0022s -  90.35%  and  88.22%
total time: 00:03:38
```

## License
© 2020 Adrian Phoulady  
This project is licensed under the MIT License.  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
