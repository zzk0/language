//
// Created by zzk on 2021/11/20.
//

#include "iostream"
#include "ModelConfig.pb.h"
#include "Address.pb.h"

int main() {
  Address address;
  std::cout << address.IsInitialized() << std::endl;
  address.set_id(999);
  std::cout << address.IsInitialized() << std::endl;
  auto map = address.mutable_id2name();
  map->operator[](19) = "asdf";
  (*map)[20] = "qwer";
  *(address.add_names()) = "good";
  *(address.add_names()) = "bad";
  auto names = address.mutable_names();
  for (auto& name : *names) {
    name += " changed";
    std::cout << name << std::endl;
  }
  std::cout << address.ShortDebugString() << std::endl;
  std::cout << address.SpaceUsed() << std::endl;
  address.Clear();
  std::cout << address.ShortDebugString() << std::endl;
  std::cout << address.SpaceUsed() << std::endl;
  std::cout << "hello world" << std::endl;
}
