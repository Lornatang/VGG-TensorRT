# VGG-TensorRT

## Overview

Use the PyTorch framework to build a network model and train the data set, and hand it over to TensorRT for inference.

### Table of contents

- [AlexNet-TensorRT](#vgg-tensorrt)
  - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [About TensorRT](#about-tensorrt)
    - [About AlexNet-TensorRT](#about-vgg-tensorrt)
    - [Installation](#installation)
      - [Clone and install requirements](#clone-and-install-requirements)
      - [Download tensorrt weights and tensorrt engine](#download-tensorrt-weights-and-tensorrt-engine)
      - [Download datasets](#download-datasets)
    - [Usage](#usage)
      - [Train](#train)
      - [Inference](#inference)
    - [Credit](#credit)

### About TensorRT

TensorRT is a C++ library for high performance inference on NVIDIA GPUs and deep learning accelerators.
More detail see [https://developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt)

### About VGG-TensorRT

This repo uses the TensorRT API to build an engine for a model trained on the [ImageNet2012 dataset](http://www.image-net.org/challenges/LSVRC/2012/).
It creates the network layer by layer, sets up weights and inputs/outputs, and then performs inference.
Both of these samples use the same model weights, handle the same input, and expect similar output.

### Installation

#### Clone and install requirements

```bash
# Install TensorRT-DevelopmentKit
wget https://github.com/Lornatang/TensorRT-Toolkit/archive/TensorRT-ToolKit-v1.1.0.tar.gz
tar xzcf TensorRT-ToolKit-v1.1.0.tar.gz
cd TensorRT-Toolkit-TensorRT-ToolKit-v1.1.0
mkdir build && cd build
cmake ..
sudo make install

# Install this repo
git clone https://github.com/Lornatang/VGG-TensorRT.git
cd VGG-TensorRT/
pip install -r requirements.txt
```

**In addition, the following conditions should also be met for TensorRT:**

- Cmake >= 3.10.2
- OpenCV >= 4.4.0
- TensorRT >= 7.0

#### Download tensorrt weights and tensorrt engine

```bash
cd weights/
bash download.sh
```

#### Download datasets

See `<AlexNet-TensorRT>/data/README.md`

### Usage

#### Train

```bash
python train.py data
```

#### Inference

1. Compile this sample by running `make` in the `<TensorRT root directory>/build` directory. The binary named `vgg` will be created in the `<TensorRT root directory>/build/bin` directory.

    ```bash
    cd <TensorRT root directory>
    mkdir build
    cd build
    cmake ..
    make
    ```

    Where `<TensorRT root directory>` is where you clone LeNet-TensorRT.

2. Run the sample to perform inference on the dog:

    ```bash
    ./vgg --engine <num_classes>// Generate TensorRT inference model.
    ./vgg --image <image_path> <num_classes> // Reasoning on the picture.
    ```

3. Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:

    ```text
    # ./vgg --image ~/Desktop/a.jpg 1000
    [INFO](14:31:3): Read from`/opt/tensorrt_models/torch/vgg/vgg.engine` inference engine.
    [INFO](14:31:5): Read image from `/home/unix/Desktop/a.jpg`!
    [INFO](14:31:5): Resize image size to 224 * 224.
    [INFO](14:31:5): Preprocess the input image.
    [INFO](14:31:5): Inference......
    --------                       -----------
    Category                       probability
    --------                       -----------
    tench                          0.515312
    red-breasted merganser          0.42625
    hornbill                       0.334688
    lorikeet                       0.297344
    goldfinch                      0.263594
    ```
  
4. Install into the system directory(optional)

    ```bash
    sudo make install
    # Create lenet engine
    vgg --engine 1000
    # Test the picture
    vgg --image ~/Desktop/test.jpg 1000
    ```

### Credit

#### Very Deep Convolutional Networks for Large-Scale Image Recognition

*Karen Simonyan, Andrew Zisserman*

##### Abstract

In this work we investigate the effect of the convolutional network depth on its accuracy in the 
large-scale image recognition setting. Our main contribution is a thorough evaluation of networks 
of increasing depth using an architecture with very small (3x3) convolution filters, which shows 
that a significant improvement on the prior-art configurations can be achieved by pushing the depth 
to 16-19 weight layers. These findings were the basis of our ImageNet Challenge 2014 submission, 
where our team secured the first and the second places in the localisation and classification tracks 
respectively. We also show that our representations generalise well to other datasets, where they 
achieve state-of-the-art results. We have made our two best-performing ConvNet models publicly 
available to facilitate further research on the use of deep visual representations in computer vision.

[paper](https://arxiv.org/abs/1409.1556)

```text
@article{VGG,
title:{Very Deep Convolutional Networks for Large-Scale Image Recognition},
author:{Karen Simonyan, Andrew Zisserman},
journal={iclr},
year={2015}
}
```
