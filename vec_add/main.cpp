#include <stdio.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdlib.h>

#include "util.h"
#include "vec_add.h"

static void print_help(const char* prog_name) {
  printf("Usage: %s [-pvh] [-t num_threads] [-n num_iterations] M\n", prog_name);
  printf("Options:\n");
  printf("  -p : print vector data. (default: off)\n");
  printf("  -v : validate vector addition. (default: off)\n");
  printf("  -h : print this page.\n");
  printf("  -t : number of threads (default: 1)\n");
  printf("  -n : number of iterations (default: 1)\n");
  printf("   M : the number of elements. (default: 8)\n");
}

static bool print_vector = false;
static bool validation = false;
static int M = 8;
static int num_threads = 1;
static int num_iterations = 1;

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "pvht:n:")) != -1) {
    switch (c) {
      case 'p':
        print_vector = true;
        break;
      case 'v':
        validation = true;
        break;
      case 't':
        num_threads = atoi(optarg);
        break;
      case 'n':
        num_iterations = atoi(optarg);
        break;
      case 'h':
      default:
        print_help(argv[0]);
        exit(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j) {
    switch (j) {
      case 0: M = atoi(argv[i]); break;
      default: break;
    }
  }
  printf("Options:\n");
  printf("  Problem size: M = %d\n", M);
  printf("  Number of threads: %d\n", num_threads);
  printf("  Number of iterations: %d\n", num_iterations);
  printf("  Print vector: %s\n", print_vector ? "on" : "off");
  printf("  Validation: %s\n", validation ? "on" : "off");
  printf("\n");
}

int main(int argc, char **argv) {
  parse_opt(argc, argv);

  printf("Initializing... "); fflush(stdout);
  float *A, *B, *C;
  alloc_vec(&A, M);
  alloc_vec(&B, M);
  alloc_vec(&C, M);
  rand_vec(A, M);
  rand_vec(B, M);
  printf("done!\n");

  double elapsed_time_sum = 0;
  for (int i = 0; i < num_iterations; ++i) {
    printf("Calculating...(iter=%d) ", i); fflush(stdout);
    zero_vec(C, M);
    timer_start(0);
    vec_add(A, B, C, M, num_threads);
    double elapsed_time = timer_stop(0);
    printf("%f sec\n", elapsed_time);
    elapsed_time_sum += elapsed_time;
  }

  if (print_vector) {
    printf("VECTOR A:\n"); print_vec(A, M);
    printf("VECTOR B:\n"); print_vec(B, M);
    printf("VECTOR C:\n"); print_vec(C, M);
  }

  if (validation) {
    check_vec_add(A, B, C, M);
  }

  double elapsed_time_avg = elapsed_time_sum / num_iterations;
  printf("Avg. time: %f sec\n", elapsed_time_avg);
  printf("Avg. throughput: %f GFLOPS\n", M / elapsed_time_avg / 1e9);

  return 0;
}
