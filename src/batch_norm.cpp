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

#include "../include/batch_norm.h"

using namespace std;
using namespace nvinfer1;

IScaleLayer *addBatchNorm2d(nvinfer1::INetworkDefinition *network,
                                      std::map<std::string, nvinfer1::Weights> &weights, nvinfer1::ITensor &input,
                                      const std::string& lname, float eps) {

  float *gamma = (float *) weights[lname + ".weight"].values;
  float *beta = (float *) weights[lname + ".bias"].values;
  float *mean = (float *) weights[lname + ".running_mean"].values;
  float *var = (float *) weights[lname + ".running_var"].values;
  int len = weights[lname + ".running_var"].count;

  auto *scval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
  for (int i = 0; i < len; i++) { scval[i] = gamma[i] / sqrt(var[i] + eps); }
  Weights scale{DataType::kFLOAT, scval, len};

  auto *shval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
  for (int i = 0; i < len; i++) { shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps); }
  Weights shift{DataType::kFLOAT, shval, len};

  auto *pval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
  for (int i = 0; i < len; i++) { pval[i] = 1.0; }
  Weights power{DataType::kFLOAT, pval, len};

  weights[lname + ".scale"] = scale;
  weights[lname + ".shift"] = shift;
  weights[lname + ".power"] = power;
  IScaleLayer *scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
  assert(scale_1);
  return scale_1;
}