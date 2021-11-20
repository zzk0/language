//
// Created by zzk on 2021/11/20.
//

#include "iostream"
#include "ModelConfig.pb.h"

int main() {
  triton::ModelConfig model_config;
  model_config.set_id(10);
  std::cout << "hello world" << std::endl;
}
