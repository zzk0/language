//
// Created by zzk on 2021/11/19.
//

#include "iostream"
#include "address.pb.h"

int main() {
  tutorial::Person person;
  person.set_name("sss");
  person.set_id(101);
  std::cout << person.DebugString() << std::endl;
  return 0;
}
