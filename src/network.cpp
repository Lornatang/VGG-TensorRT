/*
 * Copyright (c) 2020, Lorna Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../include/vgg_network.h"

using namespace nvinfer1;
using namespace std;

static const char *WEIGHTS = "/opt/tensorrt_models/torch/vgg/vgg.wts";

// Custom create VGG11 neural network engine
ICudaEngine *create_vgg11_engine(int max_batch_size, IBuilder *builder, DataType data_type, IBuilderConfig *config,
                                 int number_classes) {
  INetworkDefinition *model = builder->createNetworkV2(0);

  // Create input tensor of shape {1, 3, 224, 224} with name INPUT_NAME
  ITensor *data = model->addInput("input", data_type, Dims3{3, 224, 224});
  assert(data);

  std::map<std::string, Weights> weightMap = load_weights(WEIGHTS);

  // Add convolution layer with 6 outputs and a 5x5 filter.
  IConvolutionLayer *conv1 = model->addConvolutionNd(*data, 64, DimsHW{3, 3}, weightMap["features.0.weight"],
                                                     weightMap["features.0.bias"]);
  assert(conv1);
  conv1->setPaddingNd(DimsHW{1, 1});
  IActivationLayer *relu1 = model->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
  assert(relu1);
  IPoolingLayer *pool1 = model->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
  assert(pool1);
  pool1->setStrideNd(DimsHW{2, 2});

  conv1 = model->addConvolutionNd(*pool1->getOutput(0), 128, DimsHW{3, 3}, weightMap["features.3.weight"],
                                  weightMap["features.3.bias"]);
  conv1->setPaddingNd(DimsHW{1, 1});
  relu1 = model->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
  pool1 = model->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
  pool1->setStrideNd(DimsHW{2, 2});

  conv1 = model->addConvolutionNd(*pool1->getOutput(0), 256, DimsHW{3, 3}, weightMap["features.6.weight"],
                                  weightMap["features.6.bias"]);
  conv1->setPaddingNd(DimsHW{1, 1});
  relu1 = model->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
  conv1 = model->addConvolutionNd(*relu1->getOutput(0), 256, DimsHW{3, 3}, weightMap["features.8.weight"],
                                  weightMap["features.8.bias"]);
  conv1->setPaddingNd(DimsHW{1, 1});
  relu1 = model->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
  pool1 = model->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
  pool1->setStrideNd(DimsHW{2, 2});

  conv1 = model->addConvolutionNd(*pool1->getOutput(0), 512, DimsHW{3, 3}, weightMap["features.11.weight"],
                                  weightMap["features.11.bias"]);
  conv1->setPaddingNd(DimsHW{1, 1});
  relu1 = model->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
  conv1 = model->addConvolutionNd(*relu1->getOutput(0), 512, DimsHW{3, 3}, weightMap["features.13.weight"],
                                  weightMap["features.13.bias"]);
  conv1->setPaddingNd(DimsHW{1, 1});
  relu1 = model->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
  pool1 = model->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
  pool1->setStrideNd(DimsHW{2, 2});

  conv1 = model->addConvolutionNd(*pool1->getOutput(0), 512, DimsHW{3, 3}, weightMap["features.16.weight"],
                                  weightMap["features.16.bias"]);
  conv1->setPaddingNd(DimsHW{1, 1});
  relu1 = model->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
  conv1 = model->addConvolutionNd(*relu1->getOutput(0), 512, DimsHW{3, 3}, weightMap["features.18.weight"],
                                  weightMap["features.18.bias"]);
  conv1->setPaddingNd(DimsHW{1, 1});
  relu1 = model->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
  pool1 = model->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
  pool1->setStrideNd(DimsHW{2, 2});

  IFullyConnectedLayer *fc1 = model->addFullyConnected(*pool1->getOutput(0), 4096, weightMap["classifier.0.weight"],
                                                       weightMap["classifier.0.bias"]);
  assert(fc1);
  relu1 = model->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
  fc1 = model->addFullyConnected(*relu1->getOutput(0), 4096, weightMap["classifier.3.weight"],
                                 weightMap["classifier.3.bias"]);
  relu1 = model->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
  fc1 = model->addFullyConnected(*relu1->getOutput(0), number_classes, weightMap["classifier.6.weight"],
                                 weightMap["classifier.6.bias"]);

  fc1->getOutput(0)->setName("label");
  model->markOutput(*fc1->getOutput(0));

  // Build engine
  builder->setMaxBatchSize(max_batch_size);
  config->setMaxWorkspaceSize(1_GiB);
  config->setFlag(BuilderFlag::kFP16);
  ICudaEngine *engine = builder->buildEngineWithConfig(*model, *config);

  // Don't need the network any more
  model->destroy();

  // Release host memory
  for (auto &mem : weightMap) { free((void *) (mem.second.values)); }

  return engine;
}

// Custom create VGG16 neural network engine
ICudaEngine *create_vgg_engine(int max_batch_size, IBuilder *builder, DataType data_type, IBuilderConfig *config,
                               int number_classes) {
  INetworkDefinition *model = builder->createNetworkV2(0);

  // Create input tensor of shape {1, 3, 224, 224} with name INPUT_NAME
  ITensor *data = model->addInput("input", data_type, Dims3{3, 224, 224});
  assert(data);

  std::map<std::string, Weights> weights = load_weights(WEIGHTS);

  // Add convolution layer with 64 outputs and a 3x3 filter.
  IConvolutionLayer *conv1 =
          model->addConvolutionNd(*data, 64, DimsHW{3, 3}, weights["features.0.weight"], weights["features.0.bias"]);
  assert(conv1);
  conv1->setPaddingNd(DimsHW{1, 1});
  IActivationLayer *relu1 = model->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
  assert(relu1);
  // Add convolution layer with 64 outputs and a 3x3 filter.
  IConvolutionLayer *conv2 = model->addConvolutionNd(*relu1->getOutput(0), 64, DimsHW{3, 3},
                                                     weights["features.2.weight"], weights["features.2.bias"]);
  assert(conv2);
  conv2->setPaddingNd(DimsHW{1, 1});
  IActivationLayer *relu2 = model->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
  assert(relu2);
  // Add max pooling layer with stride of 2x2 and kernel size of 2x2
  IPoolingLayer *pool1 = model->addPoolingNd(*relu2->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
  assert(pool1);
  pool1->setStrideNd(DimsHW{2, 2});

  // Add convolution layer with 128 outputs and a 3x3 filter.
  IConvolutionLayer *conv3 = model->addConvolutionNd(*pool1->getOutput(0), 128, DimsHW{3, 3},
                                                     weights["features.5.weight"], weights["features.5.bias"]);
  assert(conv3);
  conv3->setPaddingNd(DimsHW{1, 1});
  IActivationLayer *relu3 = model->addActivation(*conv3->getOutput(0), ActivationType::kRELU);
  assert(relu3);
  // Add convolution layer with 128 outputs and a 3x3 filter.
  IConvolutionLayer *conv4 = model->addConvolutionNd(*relu3->getOutput(0), 128, DimsHW{3, 3},
                                                     weights["features.7.weight"], weights["features.7.bias"]);
  assert(conv4);
  conv4->setPaddingNd(DimsHW{1, 1});
  IActivationLayer *relu4 = model->addActivation(*conv4->getOutput(0), ActivationType::kRELU);
  assert(relu4);
  // Add max pooling layer with stride of 2x2 and kernel size of 2x2
  IPoolingLayer *pool2 = model->addPoolingNd(*relu4->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
  assert(pool2);
  pool2->setStrideNd(DimsHW{2, 2});

  // Add convolution layer with 256 outputs and a 3x3 filter.
  IConvolutionLayer *conv5 = model->addConvolutionNd(*pool2->getOutput(0), 256, DimsHW{3, 3},
                                                     weights["features.10.weight"], weights["features.10.bias"]);
  assert(conv5);
  conv5->setPaddingNd(DimsHW{1, 1});
  IActivationLayer *relu5 = model->addActivation(*conv5->getOutput(0), ActivationType::kRELU);
  assert(relu5);
  // Add convolution layer with 256 outputs and a 3x3 filter.
  IConvolutionLayer *conv6 = model->addConvolutionNd(*relu5->getOutput(0), 256, DimsHW{3, 3},
                                                     weights["features.12.weight"], weights["features.12.bias"]);
  assert(conv6);
  conv6->setPaddingNd(DimsHW{1, 1});
  IActivationLayer *relu6 = model->addActivation(*conv6->getOutput(0), ActivationType::kRELU);
  assert(relu6);
  // Add convolution layer with 256 outputs and a 3x3 filter.
  IConvolutionLayer *conv7 = model->addConvolutionNd(*relu6->getOutput(0), 256, DimsHW{3, 3},
                                                     weights["features.14.weight"], weights["features.14.bias"]);
  assert(conv7);
  conv7->setPaddingNd(DimsHW{1, 1});
  IActivationLayer *relu7 = model->addActivation(*conv7->getOutput(0), ActivationType::kRELU);
  assert(relu7);
  // Add max pooling layer with stride of 2x2 and kernel size of 2x2
  IPoolingLayer *pool3 = model->addPoolingNd(*relu7->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
  assert(pool3);
  pool3->setStrideNd(DimsHW{2, 2});

  // Add convolution layer with 512 outputs and a 3x3 filter.
  IConvolutionLayer *conv8 = model->addConvolutionNd(*pool3->getOutput(0), 512, DimsHW{3, 3},
                                                     weights["features.17.weight"], weights["features.17.bias"]);
  assert(conv8);
  conv8->setPaddingNd(DimsHW{1, 1});
  IActivationLayer *relu8 = model->addActivation(*conv8->getOutput(0), ActivationType::kRELU);
  assert(relu8);
  // Add convolution layer with 512 outputs and a 3x3 filter.
  IConvolutionLayer *conv9 = model->addConvolutionNd(*relu8->getOutput(0), 512, DimsHW{3, 3},
                                                     weights["features.19.weight"], weights["features.19.bias"]);
  assert(conv9);
  conv9->setPaddingNd(DimsHW{1, 1});
  IActivationLayer *relu9 = model->addActivation(*conv9->getOutput(0), ActivationType::kRELU);
  assert(relu9);
  // Add convolution layer with 512 outputs and a 3x3 filter.
  IConvolutionLayer *conv10 = model->addConvolutionNd(*relu9->getOutput(0), 512, DimsHW{3, 3},
                                                      weights["features.21.weight"], weights["features.21.bias"]);
  assert(conv10);
  conv10->setPaddingNd(DimsHW{1, 1});
  IActivationLayer *relu10 = model->addActivation(*conv10->getOutput(0), ActivationType::kRELU);
  assert(relu10);
  // Add max pooling layer with stride of 2x2 and kernel size of 2x2
  IPoolingLayer *pool4 = model->addPoolingNd(*relu10->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
  assert(pool4);
  pool4->setStrideNd(DimsHW{2, 2});

  // Add convolution layer with 512 outputs and a 3x3 filter.
  IConvolutionLayer *conv11 = model->addConvolutionNd(*pool4->getOutput(0), 512, DimsHW{3, 3},
                                                      weights["features.24.weight"], weights["features.24.bias"]);
  assert(conv11);
  conv11->setPaddingNd(DimsHW{1, 1});
  IActivationLayer *relu11 = model->addActivation(*conv11->getOutput(0), ActivationType::kRELU);
  assert(relu11);
  // Add convolution layer with 512 outputs and a 3x3 filter.
  IConvolutionLayer *conv12 = model->addConvolutionNd(*relu11->getOutput(0), 512, DimsHW{3, 3},
                                                      weights["features.26.weight"], weights["features.26.bias"]);
  assert(conv12);
  conv12->setPaddingNd(DimsHW{1, 1});
  IActivationLayer *relu12 = model->addActivation(*conv12->getOutput(0), ActivationType::kRELU);
  assert(relu12);
  // Add convolution layer with 512 outputs and a 3x3 filter.
  IConvolutionLayer *conv13 = model->addConvolutionNd(*relu12->getOutput(0), 512, DimsHW{3, 3},
                                                      weights["features.28.weight"], weights["features.28.bias"]);
  assert(conv13);
  conv13->setPaddingNd(DimsHW{1, 1});
  IActivationLayer *relu13 = model->addActivation(*conv13->getOutput(0), ActivationType::kRELU);
  assert(relu13);
  // Add max pooling layer with stride of 2x2 and kernel size of 2x2
  IPoolingLayer *pool5 = model->addPoolingNd(*relu13->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
  assert(pool5);
  pool5->setStrideNd(DimsHW{2, 2});

  IFullyConnectedLayer *fc1 = model->addFullyConnected(*pool5->getOutput(0), 4096, weights["classifier.0.weight"],
                                                       weights["classifier.0.bias"]);
  relu1 = model->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
  IFullyConnectedLayer *fc2 = model->addFullyConnected(*relu1->getOutput(0), 4096, weights["classifier.3.weight"],
                                                       weights["classifier.3.bias"]);
  relu2 = model->addActivation(*fc2->getOutput(0), ActivationType::kRELU);
  IFullyConnectedLayer *fc3 = model->addFullyConnected(*relu2->getOutput(0), number_classes,
                                                       weights["classifier.6.weight"], weights["classifier.6.bias"]);

  fc3->getOutput(0)->setName("label");
  model->markOutput(*fc3->getOutput(0));

  // Build engine
  builder->setMaxBatchSize(max_batch_size);
  config->setMaxWorkspaceSize(1_GiB);
  config->setFlag(BuilderFlag::kFP16);
  ICudaEngine *engine = builder->buildEngineWithConfig(*model, *config);

  // Don't need the network any more
  model->destroy();

  // Release host memory
  for (auto &mem : weights) { free((void *) (mem.second.values)); }

  return engine;
}