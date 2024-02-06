#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include "util.h"

void handle_sig2();
void handle_sigusr();

/*
 * First, print out the process ID of this process.
 *
 * Then, set up the signal handler so that ^C causes
 * the program to print "Nice try.\n" and continue looping.
 *
 * Finally, loop forever, printing "Still here\n" once every
 * second.
 */
int main(int argc, char **argv)
{
  printf("Its pid: %d\n", getpid());
  struct timespec remaining, request = {1, 0};
  signal(2, handle_sig2);
  signal(10, handle_sigusr);
  while (1) {
    printf("Still here\n");
    nanosleep(&request, &remaining);
  }
  return 0;
}

void handle_sig2() {
  ssize_t bytes;
  const int STDOUT = 1;
  bytes = write(STDOUT, "Nice try.\n", 10);
  if (bytes != 10) {
    exit(-999);
  }
}

void handle_sigusr() {
  printf("exiting\n");
  exit(0);
}


