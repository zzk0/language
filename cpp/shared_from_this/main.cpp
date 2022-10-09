#include <iostream>
#include <memory>
#include <type_traits>

class Foo;

class Test : public std::enable_shared_from_this<Test> {
public:
  Test() {}
  ~Test() = default;
  std::shared_ptr<Test> GetTest() { return std::shared_ptr<Test>(this); }
  std::shared_ptr<Test> GetSharedTest() { return shared_from_this(); }
};

int main() {
  std::shared_ptr<Test> origin = std::make_shared<Test>();
  std::shared_ptr<Test> x = origin->GetSharedTest();

  std::cout << "origin count: " << origin.use_count() << std::endl;
  std::cout << "x count: " << x.use_count() << std::endl;
  
  std::shared_ptr<Test> y = origin->GetTest();
  std::cout << "y count: " << y.use_count() << std::endl;
  return 0;
}
