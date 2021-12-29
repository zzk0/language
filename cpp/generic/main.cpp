#include "iostream"

namespace rocket {
  struct Apple {
    int a;
  };
  void initialize(Apple& apple) {
    apple.a = 10;
  }
  Apple operator+(const Apple& a, const Apple& b) {
    Apple c;
    initialize(c);
    c.a = a.a + b.a;
    return c;
  }
}

int main() {
  rocket::Apple a{}, b{};
  rocket::initialize(a);
  initialize(b);   // ADL!!! 根据参数类型来查找命名空间, 但是不会扩展到父命名空间
  operator<<(std::cout, "asdf");
  std::operator<<(std::cout, "xzcv");
  const auto& c = a + b;
}
