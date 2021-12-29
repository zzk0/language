#include "iostream"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "sys/mman.h"

int main(int argc, char *argv[]) {
  unsigned char code[] = {0xb8, 0x00, 0x00, 0x00, 0x00, 0xc3};
  code[1] = 0xff;
  code[2] = 0xff;

  void *mem = mmap(NULL, sizeof(code), PROT_EXEC | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
  memcpy(mem, code, sizeof(code));
  int (*func)() = reinterpret_cast<int (*)(void)>(mem);
  std::cout << func() << std::endl;
  return 0;
}
