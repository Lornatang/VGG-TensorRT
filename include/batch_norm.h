#ifndef BATCH_NORM_H
#define BATCH_NORM_H

#include "NvInfer.h"
#include <cassert>
#include <map>
#include <cmath>

nvinfer1::IScaleLayer *addBatchNorm2d(nvinfer1::INetworkDefinition *network,
                                      std::map<std::string, nvinfer1::Weights> &weights, nvinfer1::ITensor &input,
                                      const std::string &lname, float eps);


#endif//BATCH_NORM_H
