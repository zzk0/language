#include <memory>

#include "iostream"
#include "memory"
#include "thread"

class AppleImpl {
 public:
  AppleImpl() {
    std::cout << "AppleImpl Default Constructor" << std::endl;
  }

  ~AppleImpl() {
    std::cout << "AppleImpl Default Destructor" << std::endl;
  }

  AppleImpl(AppleImpl &apple_impl) {
    std::cout << "AppleImpl Copy Constructor" << std::endl;
  }
  AppleImpl &operator=(const AppleImpl &apple_impl) {
    std::cout << "AppleImpl Copy Assign Constructor" << std::endl;
    return *this;
  }

  AppleImpl(AppleImpl &&apple_impl) noexcept {
    std::cout << "AppleImpl Move Constructor" << std::endl;
  }
  AppleImpl &operator=(AppleImpl &&apple_impl) noexcept {
    std::cout << "AppleImpl Move Assign Constructor" << std::endl;
    return *this;
  }

  void eat() {
    std::cout << "eat" << std::endl;
  }
};

class Apple {
 public:
  Apple() : apple_(new AppleImpl()) {
  }

  Apple(const Apple &apple) = default;
  Apple &operator=(const Apple &apple) = default;

  Apple(Apple &&apple) = default;
  Apple &operator=(Apple &&apple) = default;

  void eat() {

  }
 private:
  // 如果是 unique_ptr, 那么 copy ctor 不能使用默认的, 因为 unique_ptr 不支持复制
  // std::unique_ptr<AppleImpl> apple_;
  std::shared_ptr<AppleImpl> apple_;
};

class Father;
class Son;

class Father {
 public:
  ~Father() {
    std::cout << "~Father()" << std::endl;
  }
  std::shared_ptr<Son> son_ = nullptr;
};

class Son {
 public:
  ~Son() {
    std::cout << "~Son()" << std::endl;
  }
  std::shared_ptr<Father> father_ = nullptr;
};

class Graph {
 public:
  Graph() {
    std::cout << "Graph default Constructor" << std::endl;
    this->data = new int[SIZE];
    for (int i = 0; i < SIZE; i++) {
      this->data[i] = i;
    }
  }
  ~Graph() {
    std::cout << "Graph default Destructor" << std::endl;
    delete[] this->data;
  }

  Graph(const Graph& graph) {
    std::cout << "Graph Copy Constructor" << std::endl;
    this->data = new int[SIZE];
    for (int i = 0; i < SIZE; i++) {
      this->data[i] = graph.data[i];
    }
  }
  Graph& operator=(const Graph& graph) {
    std::cout << "Graph Copy Assign Constructor" << std::endl;
    // self assign 问题, 即自己赋值给自己
    if (this == &graph) {
      return *this;
    }
    for (int i = 0; i < SIZE; i++) {
      this->data[i] = graph.data[i];
    }
    return *this;
  }

  Graph(Graph&& graph) noexcept {
    std::cout << "Graph Move Constructor" << std::endl;
    this->data = graph.data;
    graph.data = nullptr;
  }
  Graph& operator=(Graph&& graph)  noexcept {
    std::cout << "Graph Move Assign Constructor" << std::endl;
    this->data = graph.data;
    graph.data = nullptr;
    return *this;
  }

 private:
  static constexpr int SIZE = 100000;
  int* data = nullptr;
};

int main() {
  // 从一个普通的对象构造指针, 并且传给 unique_ptr
  Graph graph;
  std::unique_ptr<Graph> graph_ptr;
  graph_ptr = std::make_unique<Graph>(std::move(graph));
  graph_ptr.reset();

  std::shared_ptr<Father> father1 = std::make_shared<Father>();
  father1.reset();
  // 三种智能指针对比: auto_ptr, unique_ptr, shared_ptr
  // auto_ptr 复制拷贝的时候, 会使到原来的指针失效, 变成空指针
  // unique_ptr, 建立所有权的概念, 禁止复制, 允许使用 std::move
  // shared_ptr, 引入了引用技术的概念
  std::auto_ptr<Apple> apple2(new Apple());
  std::auto_ptr<Apple> apple3 = apple2;
  if (apple2.get() == nullptr) {
    std::cout << "yes" << std::endl;
  }
  std::unique_ptr<Apple> apple(new Apple());
  std::unique_ptr<Apple> apple1 = std::move(apple);
  std::shared_ptr<Apple> apple4(new Apple());
  const std::shared_ptr<Apple> apple5(apple4);
  const std::shared_ptr<Apple> apple6 = apple5;

  // weak_ptr 使用方法, 需要和 shared_ptr 一起使用, 并且不会改变引用计数
  std::shared_ptr<Apple> apple7(new Apple());
  std::weak_ptr<Apple> apple8(apple7);
  std::cout << apple7.use_count() << std::endl;
  std::cout << apple8.use_count() << std::endl;
  apple7 = nullptr;
  if (apple8.expired()) {
    std::cout << "yes! set empty" << std::endl;
  } else {
    std::cout << "no! it shouldn't be output" << std::endl;
  }

  // weak_ptr 尝试解决的问题: 悬空指针问题, 循环引用的问题
  // 悬空指针问题, 没复现出来, weak_ptr 可以并且只能用来检测指针是否已经置空了
  Apple *apple_raw_ptr = new Apple();
  std::thread t0 = std::thread([&]() {
    std::shared_ptr<Apple> apple_ptr(apple_raw_ptr);
    std::this_thread::sleep_for(std::chrono::seconds(1));
  });

  std::thread t1 = std::thread([&]() {
    std::shared_ptr<Apple> apple_ptr(apple_raw_ptr);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    if (apple_ptr == nullptr) {
      std::cout << "empty ptr!!!" << std::endl;
    } else {
      std::cout << "not empty ptr!!!" << std::endl;
    }
  });
  t0.join();
  t1.join();

  // 循环引用问题, 需要注释前面的代码
  {
    std::cout << "----------------------" << std::endl;
    std::shared_ptr<Father> father(new Father);
    std::shared_ptr<Son> son(new Son);
    std::cout << father.use_count() << std::endl;
    std::cout << son.use_count() << std::endl;
    father->son_ = son;
    std::cout << son.use_count() << std::endl;  // 2
    son->father_ = father;
    std::cout << father.use_count() << std::endl;
    std::cout << son.use_count() << std::endl;  // 1
    std::cout << "----------------------" << std::endl;
  }

  return 0;
}
