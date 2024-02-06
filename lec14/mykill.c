#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv)
{
  if (argc != 2) {
    fprintf(stderr, "Invalid argument");
    exit(1);
  }
  int pid = atoi(argv[1]);
  kill(pid, SIGUSR1);
  return 0;
}
