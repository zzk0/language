// TODO: c++ 中的单例模式可以搞得很复杂，下面是一种比较简单且正确的写法。
// 其他写法要考虑的问题挺多的，比如要考虑多线程安全等问题。

#include "iostream"

class Singleton {
  public:
    static Singleton& getInstance() {
        static Singleton instance;
        return instance;
    }
    void Show() {
        std::cout << "Show" << std::endl;
    }
  private:
    Singleton() = default;
};

int main() {
    {
        Singleton s = Singleton::getInstance();
        s.Show();
    }

    {
        Singleton s1 = Singleton::getInstance();
        s1.Show();
    }
    return 0;
}
