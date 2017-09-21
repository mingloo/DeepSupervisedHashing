## Deep Supervised Hashing for Image Retrieval

This is non-official Torch implementation with minor modification for [Deep Supervised Hashing for Fast Image Retrieval](http://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.pdf).

The official Caffe implementation can be found [here](https://github.com/lhmRyan/deep-supervised-hashing-DSH).

### Prerequisites
- Install [Torch](http://torch.ch/docs/getting-started.html)
- Datasets
    - Install [MNIST](http://yann.lecun.com/exdb/mnist/): `luarocks install mnist`
    - Install [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist): `luarocks install https://raw.github.com/mingloo/fashion-mnist/master/rocks/fashion-mnist-scm-1.rockspec`
    - [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html): the code could directly download this dataset.
- [matio](https://github.com/soumith/matio-ffi.torch): we use matlab scripts that adopted from [here](https://cs.nju.edu.cn/lwj/code/DPSH.zip) to calculate retrieval [Mean Average Precision, mAP](https://en.wikipedia.org/wiki/Information_retrieval).

- weight-init.lua is adopted from [here](https://github.com/e-lab/torch-toolbox/blob/master/Weight-init/weight-init.lua).

### Usage

Please run the `run.sh` for training model from scratch and obtaining final hashing representation with default parameter settings for CIFAR10/MNIST/Fashion-MNIST datasets:

```
bash run.sh
```

Then copy the output result with suffix `.mat` in `output/` directory to the `scripts/` directory and run `DSH.m` to calculate the retrieval mAP.

### Results
![DSH-CIFAR10](https://raw.github.com/mingloo/DeepSupervisedHashing/master/img/DSH-CIFAR10.png)

![DSH-MNIST](https://raw.github.com/mingloo/DeepSupervisedHashing/master/img/DSH-MNIST.png)

![DSH-Fashion-MNIST](https://raw.github.com/mingloo/DeepSupervisedHashing/master/img/DSH-Fashion-MNIST.png)
