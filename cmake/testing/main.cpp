#include "bits/stdc++.h"
#include <iostream>

struct ServerCore {
  static ServerCore& inst();
  virtual void start(int argc, const char** argv) = 0;
};

struct ServerCoreImpl : ServerCore {
private:
  void start(int argc, const char** argv) override {
    std::cout << "start" << std::endl;
  };
};

ServerCore& ServerCore::inst() {
  static ServerCoreImpl inst;
  return inst;
}

int main(int argc, const char** argv) {
  ServerCore& core = ServerCore::inst();
  core.start(argc, argv);
  return 0;
}
