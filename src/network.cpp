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

static const char *WEIGHTS = "/opt/tensorrt_models/torch/vgg/vgg11.wts";

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