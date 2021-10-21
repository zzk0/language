//
// Created by zzk on 2021/10/20.
//

#include "iostream"
#include "boost/filesystem.hpp"
#include "boost/shared_ptr.hpp"

int main() {
  std::cout << "Hello Third Party Include" << std::endl;

  boost::shared_ptr<int> isp(new int(4));

  boost::filesystem::path path = "/usr/local/cuda";
  if (path.is_relative()) {
    std::cout << "Path is relative" << std::endl;
  }
  else {
    std::cout << "Path is not relative" << std::endl;
  }
}
