#include "iostream"
#include "algorithm"
#include "vector"

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

int main() {
  TRITONSERVER_InferenceRequest *request;
  NewRequest(&request);

  std::cout << *reinterpret_cast<int*>(request) << std::endl;
}
