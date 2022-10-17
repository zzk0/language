#include "iostream"
#include "algorithm"
#include "vector"

class Parent {
 public:
  virtual void Study() {}
};

class Son : public Parent {
 public:
  void Study() {}
};

class Daughter : public Parent {
 public:
  void Study() {}
};

void DynamicCastTest() {
  Parent *father = new Son;
  Parent *mather = new Daughter;
  father->Study();
  mather->Study();

  Son *son = dynamic_cast<Son*>(mather);
  std::cout << std::endl;
}

void StaticAssertEqual() {
  // 必须是编译时常量
  constexpr int x = 1;
  constexpr int y = 1;

  // 编译时判断，如果不相等会编译时报错，代码也会变红
  static_assert(x == y, "x is not equal to y");
}

void RValue() {
  int x = 5;
  int& y = x;
//  int&& z = y;

  // 常左值引用绑定到右值，非常左值引用不可绑定到右值
//  std::string& s0 = "asdf";
  const std::string& s1 = "asdf";
  const std::string& s2 = R"(asdf)";
}

template<typename T>
int CountTwos(const T& container) {
  return std::count_if(std::begin(container), std::end(container), [](int item) {
    return item == 2;
  });
}

class Apple {
 public:
  // 如果要显式转换，那么会发送错误
  /* explicit */ operator int() const {
    return 2;
  }
};

int count_if_main() {
  int arr[] = { 1, 2, 3, 4, 5, 4, 3, 2};
  std::vector<int> vec(arr, arr + 8);
  std::cout << CountTwos(arr) << std::endl;
  std::cout << CountTwos(vec) << std::endl;

  float arr1[] = { 1.0, 2.0, 3.0 };
  std::cout << CountTwos(arr1) << std::endl;

  Apple apples[] = { Apple(), Apple() };
  std::cout << CountTwos(apples) << std::endl;

  return 0;
}

struct TRITONSERVER_InferenceRequest;

void NewRequest(TRITONSERVER_InferenceRequest** request) {
  int* x = new int;
  *x = 10;

  *request = reinterpret_cast<TRITONSERVER_InferenceRequest*>(x);
}

// 结构体可以不用定义, 完全把它当成一个指针来使用就好了
int empty_struct_main() {
  TRITONSERVER_InferenceRequest *request;
  NewRequest(&request);

  std::cout << *reinterpret_cast<int*>(request) << std::endl;
  return 0;
}

template<typename T, typename U>
class Vector {
 public:
  T value;
};

template<typename T> using intVec = Vector<T, int>; // using 和模板兼容, typedef 不行
// template<typename > typedef Vector<T, int> int1Vec;  // 不支持这样的操作
typedef Vector<float, int> floatIntVec;

using MyFunc = void(int, int);  // 可读性更好, 下面的写法甚至不记得
typedef void MyFunc2(int, int);
using MyFunc3 = void(&)(int, int);
typedef void (&MyFunc1)(int, int);

template<typename T>
class Vector<T, double> {
 public:
  T val;
};

template<>
class Vector<float, double> {
 public:
  float x;
  double y;
};

void TestStdUnique() {
  auto Print = [](const std::vector<int>& arr) {
    for (const auto& x : arr) {
      std::cout << x << " ";
    }
    std::cout << std::endl;
  };

  std::vector<int> v{1, 2, 1, 1, 3, 3, 3, 4, 5, 4};
  Print(v);
  // std::unique 的作用时将 v 中连续的重复的元素只保留一个
  // 返回值 last 的含义，[last, v.end()) 的元素是无意义的
  auto last = std::unique(v.begin(), v.end());
  Print(v);
  v.erase(last, v.end());
  Print(v);
}

int main() {
  TestStdUnique();
}
