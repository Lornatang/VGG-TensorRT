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

#ifndef TENSORT_BATCH_NORM_H
#define TENSORT_BATCH_NORM_H

#include "NvInfer.h"
#include <cassert>
#include <cmath>
#include <map>

nvinfer1::IScaleLayer *addBatchNorm2d(nvinfer1::INetworkDefinition *network,
                                      std::map<std::string, nvinfer1::Weights> &weights, nvinfer1::ITensor &input,
                                      const std::string &lname, float eps);


#endif// TENSORT_BATCH_NORM_H
