#include <future>
#include "iostream"
#include "functional"
#include "queue"
#include "thread"
#include "mutex"
#include "condition_variable"

// c++ 多线程编程概念: https://changkun.de/modern-cpp/zh-cn/07-thread/index.html
// thread, mutex, unique_lock, lock_guard, future, packaged_task
// conditional_variable, atomic, volatile,

// 线程池就是一个生产者消费者模型
// 执行过程发生了异常或者死锁
// 线程池需要退出死循环, 设计一个 join 方法,
// 析构函数删除 thread 对象
// 代码的最后有一些 note, 随便看看就好

bool Check() {
  std::cout << "check" << std::endl;
  return false;
}

template<int N>
class ThreadPool {
 public:
  ThreadPool() {
    for (int i = 0; i < N; i++) {
      threads_[i] = std::thread([this]() {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(mutex_);
            // 在等待的时候会释放 lock, 被唤醒之后获取 lock
            cv_.wait(lock, [this]() { return Check() || !queue_.empty() || stop_; });
            if (stop_ && queue_.empty()) {
              return;
            }
            task = std::move(queue_.front());
            queue_.pop();
          }
          task();
        }
      });
    }
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto &worker : threads_) {
      worker.join();
    }
  }

  void Enqueue(std::function<void()> func) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (stop_) {
      throw std::runtime_error("submit on stopped ThreadPool");
    }
    queue_.emplace(func);
    cv_.notify_one();
  }

  template<typename F, typename... Args>
  auto Submit(F &&f, Args &&... args) -> std::future<decltype(f(args...))> {
    auto taskPtr = std::make_shared<std::packaged_task<decltype(f(args...))()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    std::function<void()> task = [taskPtr]() {
      (*taskPtr)();
    };
    Enqueue(task);
    return taskPtr->get_future();
  }

 private:
  std::queue<std::function<void()>> queue_;
  std::thread threads_[N];
  std::mutex mutex_;  // lock the queue_
  std::condition_variable cv_;  // control the thread_ status
  bool stop_ = false;
};

int main() {
  ThreadPool<4> pool;
  for (int i = 0; i < 100; i++) {
    pool.Enqueue([i]() {
      std::cout << i << " " << std::this_thread::get_id() << std::endl;
    });
  }

  std::future<int> future = pool.Submit([]() -> int {
    std::cout << std::this_thread::get_id() << std::endl;
    return 666;
  });
  std::cout << future.get() << std::endl;

//  std::this_thread::sleep_for(std::chrono::seconds(3));
  std::cout << "Main End" << std::endl;
  return 1;
}

// note 1. terminate called without an active exception
// 发生这个问题的原因是主线程退出了, 但是主线程上面开的线程还没有退出
// 解决办法是
// - join, 当前线程会进入等待状态
// - detach, 两个线程会分离, 开的线程会在后台运行了, 看不到输出了
//
//
//std::thread thread([]() {
//  for (int i = 0; i < 10; i++) {
//    std::this_thread::sleep_for(std::chrono::seconds(1));
//    std::cout << "ticking... " << i << std::endl;
//  }
//});
//
//std::this_thread::sleep_for(std::chrono::seconds(3));
//std::cout << "Main End" << std::endl;

// note2. 实现的时候发现并没有输出想要的个数
// 我猜想的可能的原因是 所有的 worker 都在跑, 某些 notify 没有起作用,
// 所以我们可以看到 queue_ 中会有多的任务
// 如果没有可以唤醒的线程, notify_one 并不会阻塞

// note3. 一开始我对 conditional_wait 的不理解, 第二个参数的作用是什么
// 第二个参数的作用, 我猜应该是在进入休眠之前的检查, 如果 fail 了, 那么进入
// 休眠, 否则将不会 wait. 我们可以看到上面的 Check 函数, 如果是轮询,
// 那么会一直输出 check 才对
