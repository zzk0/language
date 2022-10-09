#include <bits/types/FILE.h>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <sched.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>

constexpr int STACK_SIZE = 1024 * 1024;

static char child_stack[STACK_SIZE];
char* const child_args[] = {
    "/bin/bash",
    NULL
};

int child_main(void* args) {
    std::cout << "child process " << geteuid() << " " << getegid() << std::endl;
    sethostname("NewHostName", 12);
    execv(child_args[0], child_args);
    return 1;
}

int main() {
    // root permission is required
    int child_pid = clone(child_main, child_stack + STACK_SIZE,
                          CLONE_NEWUTS | SIGCHLD, NULL);
    std::cout << child_pid << " " << errno << std::endl;
    waitpid(child_pid, NULL, 0);
    std::cout << "main exited" << std::endl;
    return 0;
}
