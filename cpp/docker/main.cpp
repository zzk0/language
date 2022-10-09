#include <bits/types/FILE.h>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <sched.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/capability.h>

constexpr int STACK_SIZE = 1024 * 1024;

static char child_stack[STACK_SIZE];
char* const child_args[] = {
    "/bin/bash",
    NULL
};

void set_uid_map(pid_t pid, int inside_id, int outside_id, int length) {
    char path[256];
    sprintf(path, "/proc/%d/uid_map", getpid());
    FILE* uid_map = fopen(path, "w");
    fprintf(uid_map, "%d %d %d", inside_id, outside_id, length);
    fclose(uid_map);
}

void set_gid_map(pid_t pid, int inside_id, int outside_id, int length) {
    char path[256];
    sprintf(path, "/proc/%d/gid_map", getpid());
    FILE* gid_map = fopen(path, "w");
    fprintf(gid_map, "%d %d %d", inside_id, outside_id, length);
    fclose(gid_map);
}

int child_main(void* args) {
    std::cout << "child process " << geteuid() << " " << getegid() << std::endl;
    set_uid_map(getpid(), 0, 1008, 1);  // id -u => 1008
    set_gid_map(getpid(), 0, 1008, 1);  // id -g => 1008
    cap_t caps = cap_get_proc();
    std::cout << cap_to_text(caps, NULL) << std::endl;
    execv(child_args[0], child_args);
    return 1;
}

int main() {
    int child_pid = clone(child_main, child_stack + STACK_SIZE,
                          CLONE_NEWUSER | SIGCHLD, NULL);
    std::cout << child_pid << " " << errno << std::endl;
    waitpid(child_pid, NULL, 0);
    std::cout << "main exited" << std::endl;
    return 0;
}
