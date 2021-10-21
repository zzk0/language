#include "iostream"
#include "queue"
#include "thread"
#include "mutex"
#include "condition_variable"
#include "functional"

int main() {
  std::queue<std::function<void()>> produced_nums;
  std::mutex mutex;
  std::condition_variable cv;
  bool notified = false;

  auto producer = [&]() {
    for (int i = 0;; i++) {
      std::this_thread::sleep_for(std::chrono::milliseconds(900));
      std::unique_lock<std::mutex> lock(mutex);
      std::cout << "producing " << i << std::endl;
      produced_nums.push([i]() {
        std::cout << "consuming " << i << std::endl;
      });
      notified = true;
      cv.notify_all();
    }
  };

  auto consumer = [&]() {
    while (true) {
      std::unique_lock<std::mutex> lock(mutex);
      while (!notified) {
        cv.wait(lock);
      }

      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      lock.lock();
      while (!produced_nums.empty()) {
//        std::cout << "consuming " << produced_nums.front() << std::endl;
        produced_nums.front()();
        produced_nums.pop();
      }
      notified = false;
    }
  };

  std::thread p(producer);
  std::thread c0(consumer);
  std::thread c1(consumer);
  p.join();
  c0.join();
  c1.join();

  return 0;
}
