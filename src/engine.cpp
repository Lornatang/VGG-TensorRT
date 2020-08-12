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

#include "../include/vgg_engine.h"

using namespace nvinfer1;
using namespace std;

static Logger gLogger; /* NOLINT */

void serialize_vgg_engine(int max_batch_size, IHostMemory **model_stream, int number_classes) {
  // Create builder
  report_message(0);
  cout << "Creating builder..." << endl;
  IBuilder *builder = createInferBuilder(gLogger);
  IBuilderConfig *config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an engine
  report_message(0);
  cout << "Creating VGG16 network engine..." << endl;
  ICudaEngine *engine = create_vgg_engine(max_batch_size, builder, DataType::kFLOAT, config, number_classes);

  assert(engine != nullptr);

  // Serialize the engine
  report_message(0);
  cout << "Serialize model engine..." << endl;
  (*model_stream) = engine->serialize();

  // Close everything down
  engine->destroy();
  builder->destroy();
}