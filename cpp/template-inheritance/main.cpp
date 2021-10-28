//
// Created by zzk on 2021/10/26.
//

#include "iostream"
#include "vector"
#include "memory"

class Tensor {

};

class TensorTuple final : std::vector<std::shared_ptr<Tensor>>,
                          std::enable_shared_from_this<TensorTuple> {

};

int main() {
  std::cout << "asdf" << std::endl;
}
