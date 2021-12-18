//
// Created by zzk on 2021/11/17.
//

#include "iostream"
#include "vector"

class Apple {
 public:
  Apple() {
    // this 指针指向这个类的首地址, 如果 new 一个类, 他们的地址是一样的
    std::cout << this << std::endl;
    std::cout << "Apple Default Constructor" << std::endl;
  }
  explicit Apple(int i) {
    std::cout << "Apple Parameter Constructor" << std::endl;
    this->data = i;
  }
  ~Apple() {
    std::cout << "Apple Default Destructor" << std::endl;
  }
//  Apple(const Apple &apple) = delete;
//  Apple(const Apple &&apple) = delete;

  Apple(const Apple &apple) {
    std::cout << "Apple Copy Constructor" << std::endl;
  }
  Apple &operator=(const Apple &apple) {
    std::cout << "Apple Copy Assign Constructor" << std::endl;
    return *this;
  }
  Apple(Apple &&apple) noexcept {
    std::cout << "Apple Move Constructor" << std::endl;
  }
  Apple &operator=(Apple &&apple) noexcept {
    std::cout << "Apple Move Assign Constructor" << std::endl;
    return *this;
  }
 private:
  int data = 0;
};

//Apple getApple() {
//  Apple apple(10);
//  return apple;
//}

void SetPtr(int *&ptr) {
  ptr = new int(10);
}

int main() {
  std::allocator<Apple> allocator;
  Apple *apples = allocator.allocate(2);

  Apple apple0;
  Apple apple1(233);

  allocator.construct(apples, std::move(apple0));
  allocator.construct(apples + 1, apple1);

  return 1;
}

/*
如果全部都声明, 正常输出:
Apple Parameter Constructor
Apple Move Constructor
Apple Default Destructor
Apple Default Destructor

如果声明复制构造函数, 但是没有移动构造函数: 这意味着移动构造函数不会生成了
Apple Parameter Constructor
Apple Copy Constructor
Apple Default Destructor
Apple Default Destructor

如果声明移动构造函数, 但是没有复制构造函数, 那么不会生成复制构造函数, 以下操作不存在:
Apple apple = getApple();
Apple apple1;
apple1 = apple;  // implicitly deleted
*/
