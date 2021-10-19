// reference: https://mocuishle0.github.io/post/c11-xin-te-zheng-ke-bian-can-shu-mo-ban-variadic-template/

#include <tuple>
#include "iostream"

template<typename T>
void print(const T& first) {
  std::cout << first << std::endl;
}

template<typename FirstArg, typename... T>
void print(const FirstArg& first, const T&... args) {
  std::cout << first << " ";
  print(args...);
}

template<typename... T>
void count(const T&... elements) {
  std::cout << sizeof...(elements) << std::endl;
}

// 展开可变模板参数函数的参数包的方法有二:
// 1. 使用递归函数, 正如上面的 print 一样
// 2. 使用逗号表达式来展开参数包

template<typename T>
void printArg(const T& t) {
  std::cout << t << " ";
}

template<typename ...Args>
void expand(Args... args) {
  int unused[] = {(printArg(args), 0)...};
  // 关键是利用了逗号表达式: d = (a = b, c), d 的结果是 c, 不过仍然会执行 a = b 这个语句
  // 然后利用初始化列表来初始化一个可变的数组
  // 虽然还没理解为什么 ... 要加到那里, 把它当成一种语法格式就好了, 不要太在意
  std::cout << std::endl;
}

int main() {
  count();
  count('a', 1, 3, "asdf", "xxxx");
  print('a', 1, 3, "asdf", "xxxx");
  expand('a', 1, 3, "asdf", "xxxx");


  std::tuple<> tp;
  std::tuple<int> tp1 = std::make_tuple(1);
  std::tuple<int, double> tp2 = std::make_tuple(1, 2.5);
  std::tuple<int, float> tp3 = {1, 3.1f};
}
