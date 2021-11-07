#include "iostream"
#include "thread"
#include "mutex"
#include "future"
#include "vector"
#include "condition_variable"

/*
Concept

thread: 初始化即运行, 使用 join, 避免主线程退出导致的 unexpected terminated
mutex: 互斥量, 互斥量为 1, 提供了 lock unlock 成员函数
lock_guard: 自动调用 mutex 的 lock unlock 函数, 即使异常发生也可以 unlock
            它的实现很简单, 就构造函数加锁和析构函数解锁, explicit 需要显式转换, 删掉复制构造函数
unique_lock: 独占 mutex 所有权(没理解);  提供了 lock 和 unlock 方法
future: 获取异步操作的结果, 对象的获取一般是来自于别的类, 提供 wait, get 方法等待和获取结果
packaged_task: 分装函数
conditional_variable: 条件变量, 提供了 wait 方法
atomic: 指令级别的原子操作, 这是个泛型类, 但是并非所有的类型都支持原子操作, 使用 is_lock_free 检查

一致性模型: https://changkun.de/modern-cpp/zh-cn/07-thread/index.html
- 线性一致性: 每一次读操作都要能读到最新的写, atomic, 其他操作要按照顺序执行
- 顺序一致性: 每一次读操作都要能读到最新的写, 不严格要求其他操作的顺序, 即最新操作能读到就行, 其他随便
- 因果一致性: 有因果关系的操作需要保证顺序, 其他不用
- 最终一致性: 某操作可以在未来被读到
*/

static int counter = 0;
static std::atomic<int> atomic_counter = 0;

class AtomicBasedLock {
 public:
  AtomicBasedLock() = default;
  ~AtomicBasedLock() = default;

  AtomicBasedLock(const AtomicBasedLock &) = delete;
  AtomicBasedLock &operator=(const AtomicBasedLock &) = delete;

  void Lock() {
    while (flag_.test_and_set(std::memory_order_acquire)) {

    }
  }

  void Unlock() {
    flag_.clear(std::memory_order_release);
  }

 private:
//  std::atomic<bool> mutex { false };
  std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

std::atomic_flag global_flag = ATOMIC_FLAG_INIT;
AtomicBasedLock lock;

void Tick() {
  lock.Lock();
  counter += 1;
  lock.Unlock();
}

void critical_section(int value) {
  static std::mutex mtx, mtx1;
// std::lock_guard<std::mutex> lock(mtx);
// std::unique_lock<std::mutex> lock(mtx);
  std::scoped_lock<std::mutex, std::mutex> lock(mtx, mtx1);
  std::condition_variable cv;
  counter += value;
}

void MemoryRelaxed() {
  std::atomic<int> counter = {0};
  std::vector<std::thread> vt;
  vt.reserve(100);
  for (int i = 0; i < 100; ++i) {
    vt.emplace_back([&](){
      counter.fetch_add(1, std::memory_order_relaxed);
    });
  }

  for (auto& t : vt) {
    t.join();
  }
  std::cout << "current counter:" << counter << std::endl;
}

void MemoryReleaseConsume() {
  // 初始化为 nullptr 防止 consumer 线程从野指针进行读取
  std::atomic<int*> ptr(nullptr);
  int v;
  std::thread producer([&]() {
    int* p = new int(42);
    v = 1024;
    ptr.store(p, std::memory_order_release);
  });
  std::thread consumer([&]() {
    int* p;
    while(!(p = ptr.load(std::memory_order_consume)));

    std::cout << "p: " << *p << std::endl;
    std::cout << "v: " << v << std::endl;
  });
  producer.join();
  consumer.join();
}

int main() {
  for (int k = 0; k < 1000; k++) {
    std::vector<std::thread> threads;
    threads.reserve(100);
    for (int i = 0; i < 100; i++) {
      threads.emplace_back(std::thread(Tick));
    }
    for (int i = 0; i < 100; i++) {
      threads[i].join();
    }
    if (counter != 100) {
      std::cout << "Failed " << counter << " " << k << std::endl;
      return -1;
    }
    counter = 0;
  }
  std::cout << "Passed" << std::endl;

//  std::thread t1(critical_section, 3), t2(critical_section, 4);
//  t1.join();
//  t2.join();
//  std::cout << counter << std::endl;

  return 0;
}
