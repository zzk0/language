//
// Created by zzk on 2021/10/26.
//

#include "iostream"
#include "vector"
#include "memory"

class Apple {
public:
  Apple() {
    std::cout << "construct" << std::endl;
  }

  ~Apple() {
    std::cout << "desctruct" << std::endl;
  }
};

int main() {
  std::unique_ptr<Apple> apple;
  apple.reset(new Apple());
  apple.reset();
}
