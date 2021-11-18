//
// Created by zzk on 2021/11/17.
//

#include "iostream"
#include "memory"

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
    this->data = i;
  }
  ~Apple() {
    std::cout << "Apple Default Destructor" << std::endl;
  }
  Apple(const Apple& apple) {
    std::cout << "Apple Copy Constructor" << std::endl;
  }
  Apple& operator=(const Apple& apple) {
    std::cout << "Apple Copy Assign Constructor" << std::endl;
    return *this;
  }
  Apple(Apple&& apple) noexcept {
    std::cout << "Apple Move Constructor" << std::endl;
  }
  Apple& operator=(Apple&& apple) noexcept {
    std::cout << "Apple Move Assign Constructor" << std::endl;
    return *this;
  }
 private:
  int data = 0;
};

Apple getApple() {
  Apple apple(10);
  return apple;
}

int main() {
  Apple apple = getApple();

  // Graph 中的 SetNodeData 方法是 const 的, 可以使用成员的非 const 方法
//  Graph graph;
//  std::cout << graph.GetNodeData() << std::endl;
//  graph.SetNodeData(10);
//  std::cout << graph.GetNodeData() << std::endl;
  return 1;
}
