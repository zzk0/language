#include "iostream"
#include "memory"
#include "unordered_map"

 class Apple : public std::enable_shared_from_this<Apple> {
 public:
  void func() {
    std::shared_ptr<Apple> apple_ptr = shared_from_this();
    std::cout << apple_ptr.use_count() << std::endl;
  }
};


int main() {
  std::shared_ptr<Apple> apple1(new Apple());
  std::cout << apple1.use_count() << std::endl;
  apple1->func();
  std::cout << apple1.use_count() << std::endl;
  apple1->func();
  std::cout << apple1.use_count() << std::endl;
  return 1;
}
