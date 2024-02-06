#include <cstdio>
#include <random>
#include <complex>
#include <sys/time.h>
#include <immintrin.h>

const double PI = 3.1415926535897932384626433832;

static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

struct Complex {
	double real = 0;
	double imag = 0;
};

Complex* buf;

void Shuffle(Complex* arr, int n) {
  // Copy odd elements to buffer
	for (int i = 0; i < n / 2; ++i) {
		buf[i] = arr[i * 2 + 1];
  }
  // Move even elements to front
	for (int i = 0; i < n / 2; i++) {
    arr[i] = arr[i * 2];
  }
  // Copy odd elements back to end
	for (int i = 0; i < n / 2; i++) {
    arr[i + n / 2] = buf[i];
  }
}

void FFT(Complex* arr, int n) {
	if (n < 2) return;
  Shuffle(arr, n);
  FFT(arr, n / 2);
  FFT(arr + n / 2, n / 2);
  for (int k = 0; k < n / 2; k++) {
    Complex e = arr[k];
    Complex o = arr[k + n / 2];
    Complex w;
    w.real = cos(-2.0 * PI * k / n);
    w.imag = sin(-2.0 * PI * k / n);
    Complex wo;
    wo.real = w.real * o.real - w.imag * o.imag;
    wo.imag = w.real * o.imag + w.imag * o.real;
    arr[k].real = e.real + wo.real;
    arr[k].imag = e.imag + wo.imag;
    arr[k + n / 2].real = e.real - wo.real;
    arr[k + n / 2].imag = e.imag - wo.imag;
  }
}

void FFT_SIMD(Complex* arr, int n) {
	if (n <= 4) {
    FFT(arr, n);
    return;
  }
  Shuffle(arr, n);
  FFT_SIMD(arr, n / 2);
  FFT_SIMD(arr + n / 2, n / 2);
  for (int k = 0; k < n / 2; k += 2) {
    // First goal:
    // wo.real = w.real * o.real - w.imag * o.imag;
    // wo.imag = w.real * o.imag + w.imag * o.real;

    // 1. Load o = (o.real, o.imag)
    __m256d o = _mm256_load_pd((double *)&arr[k + n / 2]);

    // 2. Construct wr = (w.real, w.real)
    //          and wi = (w.imag, w.imag)
    double cos0 = cos(-2.0 * PI * (k + 0) / n);
    double sin0 = sin(-2.0 * PI * (k + 0) / n);
    double cos1 = cos(-2.0 * PI * (k + 1) / n);
    double sin1 = sin(-2.0 * PI * (k + 1) / n);
    __m256d wr = _mm256_set_pd(cos1, cos1, cos0, cos0);
    __m256d wi = _mm256_set_pd(sin1, sin1, sin0, sin0);
    
    // 3. Compute (w.real * o.real, w.real * o.imag)
    wr = _mm256_mul_pd(wr, o);

    // 4. Shuffle o = (o.real, o.imag)
    //        and get (o.imag, o.real)
    // __m256d n1 = _mm256_set_pd(o[2], o[3], o[0], o[1]);
    // __m256d n1 = _mm256_shuffle_pd(o, o, 0b0101);
    __m256d n1 = _mm256_permute_pd(o, 0b0101);

    // 5. Compute (w.imag * o.imag, w.imag * o.real)
    wi = _mm256_mul_pd(wi, n1);

    // 6. Compute (w.real * o.real - w.imag * o.imag, ...)
    n1 = _mm256_sub_pd(wr, wi);

    // 7. Compute (..., w.real * o.imag + w.imag * o.real)
    wr = _mm256_add_pd(wr, wi);

    // 8. Merge the first of n1 and the second of wr
    n1 = _mm256_shuffle_pd(n1, wr, 0b1010);

    // Easy remaining parts...
    o = _mm256_load_pd((double *)&arr[k]);
    wr = _mm256_add_pd(o, n1);
    wi = _mm256_sub_pd(o, n1);
    _mm256_store_pd((double *)&arr[k], wr);
    _mm256_store_pd((double *)&arr[k + n / 2], wi);
  }
}

int main() {
  std::default_random_engine generator(42);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  const int NELEM = 32768;
  const int NITER = 10;
  double* a = (double*)aligned_alloc(32, sizeof(double) * NELEM);
  double* b = (double*)aligned_alloc(32, sizeof(double) * NELEM);
  double* c = (double*)aligned_alloc(32, sizeof(double) * NELEM);
  Complex* ca = (Complex*)aligned_alloc(32, sizeof(Complex) * NELEM);
  Complex* cb = (Complex*)aligned_alloc(32, sizeof(Complex) * NELEM);
  Complex* cc = (Complex*)aligned_alloc(32, sizeof(Complex) * NELEM);
  buf = (Complex*)aligned_alloc(32, sizeof(Complex) * NELEM);

  for (int i = 0; i < NELEM; ++i) {
    a[i] = distribution(generator);
    b[i] = distribution(generator);
  }

  double st, et;

  // 1. Direct convolution
  /*
  st = get_time();
  for (int l = 0; l < NITER; ++l) {
    for (int i = 0; i < NELEM; ++i) {
      c[i] = 0;
      for (int j = 0; j < NELEM; ++j) {
        c[i] += a[j] * b[(i - j + NELEM) % NELEM];
      }
    }
  }
  et = get_time();
  printf("Direct convolution: %lf sec\n", et - st);
  */


  // 3. Convolution with FFT (SIMD)
  st = get_time();
  for (int l = 0; l < NITER; ++l) {
    for (int i = 0; i < NELEM; ++i) {
      ca[i].real = a[i];
      ca[i].imag = 0;
      cb[i].real = b[i];
      cb[i].imag = 0;
    }
    FFT_SIMD(ca, NELEM);
    FFT_SIMD(cb, NELEM);
    for (int i = 0; i < NELEM; ++i) {
      cc[i].real = ca[i].real * cb[i].real - ca[i].imag * cb[i].imag;
      cc[i].imag = ca[i].real * cb[i].imag + ca[i].imag * cb[i].real;
    }
    FFT_SIMD(cc, NELEM);
  }
  et = get_time();
  printf("FFT (SIMD) convolution: %lf sec\n", et - st);

  // 2. Convolution with FFT
  st = get_time();
  for (int l = 0; l < NITER; ++l) {
    for (int i = 0; i < NELEM; ++i) {
      ca[i].real = a[i];
      ca[i].imag = 0;
      cb[i].real = b[i];
      cb[i].imag = 0;
    }
    FFT(ca, NELEM);
    FFT(cb, NELEM);
    for (int i = 0; i < NELEM; ++i) {
      cc[i].real = ca[i].real * cb[i].real - ca[i].imag * cb[i].imag;
      cc[i].imag = ca[i].real * cb[i].imag + ca[i].imag * cb[i].real;
    }
    FFT(cc, NELEM);
  }
  et = get_time();
  printf("FFT convolution: %lf sec\n", et - st);

  // 4. Compare
  /*
  int err_cnt = 0, err_threshold = 10;
  for (int i = 0; i < NELEM; ++i) {
    double expected = c[i];
    double actual = cc[(NELEM - i) % NELEM].real / NELEM;
    if (fabs(expected - actual) > 1e-6) {
      ++err_cnt;
      if (err_cnt <= err_threshold) {
        printf("Error at %d: expected %lf, actual %lf\n", i, expected, actual);
      }
      if (err_cnt == err_threshold + 1) {
        printf("Too many errors. Stop printing error messages.\n");
        exit(1);
      }
    }
  }
  printf("Result: VALID\n");
  */

  free(a); free(b); free(c);
  free(ca); free(cb); free(cc);

  return 0;
}