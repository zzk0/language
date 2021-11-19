//
// Created by zzk on 2021/11/17.
//

#include "iostream"
#include "memory"
#include "unordered_map"

class Node {
 public:
  Node() : data_(0) {}
  void SetData(int data) {
    data_ = data;
  }
  int GetData() const {
    return data_;
  }
 private:
  int data_;
};

class Graph {
 public:
  Graph() {
    node_ = std::make_unique<Node>();
  }
  void SetNodeData(int data) const {
    node_->SetData(data);
  }
  int GetNodeData() const {
    return node_->GetData();
  }
 private:
  std::unique_ptr<Node> node_;
};

class Apple {
 public:
  Apple() {
    std::cout << "Apple Default Constructor" << std::endl;
  }
  explicit Apple(int i) {
    std::cout << "Apple Parameter Constructor" << std::endl;
    this->data_ = i;
  }
  ~Apple() {
    std::cout << "Apple Default Destructor" << std::endl;
  }
  Apple(const Apple &apple) {
    this->data_ = apple.data_;
    std::cout << "Apple Copy Constructor" << std::endl;
  }
  Apple &operator=(const Apple &apple) {
    this->data_ = apple.data_;
    std::cout << "Apple Copy Assign Constructor" << std::endl;
    return *this;
  }
  Apple(Apple &&apple) noexcept {
    this->data_ = apple.data_;
    std::cout << "Apple Move Constructor" << std::endl;
  }
  Apple &operator=(Apple &&apple) noexcept {
    this->data_ = apple.data_;
    std::cout << "Apple Move Assign Constructor" << std::endl;
    return *this;
  }
  int GetData() const {
    return data_;
  }
  bool operator==(const Apple& apple) const {
    return this->data_ == apple.data_;
  }

 private:
  int data_ = 0;
};

Apple getApple() {
  Apple apple(10);
  return apple;
}

template<>
struct std::hash<Apple> {
  std::size_t operator()(const Apple& apple) const {
    return std::hash<int>()(apple.GetData());
  }
};

//template<>
//struct std::equal_to<Apple> {
//  bool operator()(const Apple& x, const Apple& y) const {
//    return x.GetData() == y.GetData();
//  }
//};

struct apple_hash {
  std::size_t operator()(const Apple& apple) const {
    return std::hash<int>()(apple.GetData());
  }
};

//struct apple_equal {
//  bool operator()(const Apple& x, const Apple& y) const {
//    std::cout << "apple_equal" << std::endl;
//    return x.GetData() == y.GetData();
//  }
//};

template<typename T, typename U>
void PrintHashInfo(std::unordered_map<T, U> hash) {
  std::cout << "load factor: " << hash.load_factor() << std::endl;
  std::cout << "max load factor: " << hash.max_load_factor() << std::endl;
  std::cout << "bucket count: " << hash.bucket_count() << std::endl;
  std::cout << "max bucket count: " << hash.max_bucket_count() << std::endl;
  std::cout << "size: " << hash.size() << std::endl;
  std::cout << "max size: " << hash.max_size() << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
}

int main() {
  std::unordered_map<int, int> map;
  PrintHashInfo(map);
  for (int i = 0; i < 1000; i++) {
    map[i] = i;
    PrintHashInfo(map);
  }

  // Graph 中的 SetNodeData 方法是 const 的, 可以使用成员的非 const 方法
//  Graph graph;
//  std::cout << graph.GetNodeData() << std::endl;
//  graph.SetNodeData(10);
//  std::cout << graph.GetNodeData() << std::endl;
  return 1;
}
