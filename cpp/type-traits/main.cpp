// https://en.cppreference.com/w/cpp/header/type_traits

#include "iostream"

namespace util {

// 声明为静态变量的原因：is_void<int>::value
// 为了可以像上面那样去使用
template<typename T>
struct is_void {
  static const bool value = false;
};

// 特化，全特化
template<>
struct is_void<void> {
  static const bool value = true;
};

template<typename T, typename U>
struct is_same {
  const bool value;
};

template<typename T>
struct is_integral {
  is_integral() {
    value = std::is_same<int, T>::value;
  }
  const bool value;
};

template<typename Tp, Tp _v>
struct integral_constant {
  static constexpr Tp value = _v;
};

}

int main() {
  std::cout << std::is_integral<float>::value << std::endl;
  std::cout << std::is_integral<double>::value << std::endl;
  std::cout << std::is_integral<char>::value << std::endl;
  std::cout << std::is_integral<short>::value << std::endl;
  std::cout << std::is_integral<int>::value << std::endl;
  std::cout << std::is_integral<long>::value << std::endl;
  std::is_same<int, int>::value;
  std::is_same<std::conditional<true, int, double>, int>::value;

  std::cout << util::is_void<int>::value << std::endl;
  std::cout << util::is_void<void>::value << std::endl;

  std::cout << std::integral_constant<int, 10>::value << std::endl;
  std::integral_constant<long, 100> z;
  std::true_type x;
  std::false_type y;
  std::cout << x << " " << y << " " << z << std::endl;

  typedef std::integral_constant<int, 2> two_t;
  two_t x0;
  std::cout << x0 << std::endl;

  std::cout << util::integral_constant<int, 233>::value << std::endl;
  util::integral_constant<bool, true> true_value_;
  typedef util::integral_constant<bool, false> my_false_type;
  my_false_type mx;
}

