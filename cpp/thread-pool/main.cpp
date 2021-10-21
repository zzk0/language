#include "iostream"
#include "functional"
#include "queue"
#include "thread"
#include "mutex"
#include "condition_variable"

// 线程池就是一个生产者消费者模型
// 执行过程发生了异常
// 线程池需要退出死循环, 设计一个 join 方法, 和析构函数

template<int N>
class ThreadPool {
 public:
  ThreadPool() {
    for (int i = 0; i < N; i++) {
      threads_[i] = std::thread([&]() {
        while (true) {
          std::unique_lock<std::mutex> lock(mutex_);
          while (!notified_) {
            // 在等待的时候会释放 lock, 被唤醒之后获取 lock
            cv_.wait(lock);
          }

          if (!queue_.empty()) {
            std::function<void()> func = queue_.front();
            queue_.pop();
            func();
            notified_ = false;
          }
        }
      });
    }
  }
  ~ThreadPool() {}

  void Enqueue(std::function<void()> func) {
    std::unique_lock<std::mutex> lock(mutex_);
    queue_.push(func);
    cv_.notify_one();
    notified_ = true;
  }

 private:
  std::queue<std::function<void()>> queue_;
  std::thread threads_[N];
  std::mutex mutex_;  // lock the queue_
  std::condition_variable cv_;  // control the thread_ status
  bool notified_ = false;
  bool exit_ = false;
};


int main() {
  ThreadPool<4> pool;
  for (int i = 0; i < 100; i++) {
    pool.Enqueue([i]() {
      std::cout << i << " " << std::this_thread::get_id() << std::endl;
    });
  }

  return 1;
}
