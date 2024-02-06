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

void Shuffle(Complex* arr, int n) {
  Complex* buf = new Complex[n / 2];
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
	delete[] buf;
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
  if (n < 2) return;
  Shuffle(arr, n);
  FFT_SIMD(arr, n / 2);
  FFT_SIMD(arr + n / 2, n / 2);

  __m256d ereal, eimag, oreal, oimag;
  __m256d wreal, wimag;
  __m256d w0real, w0imag;
  __m256d freal, fimag;
  __m256d sreal, simag;

  for (int i = 0; i < n / 2; i += 4) {
    
    double rtemp1, rtemp2, rtemp3, rtemp4, rtemp5, rtemp6, rtemp7, rtemp8;
    double itemp1, itemp2, itemp3, itemp4, itemp5, itemp6, itemp7, itemp8;

    rtemp1 = arr[i].real;
    itemp1 = arr[i].imag;
    if (i + 1 < n / 2) {
      rtemp2 = arr[i + 1].real;
      rtemp5 = arr[i + n / 2 + 1].real;
      itemp2 = arr[i + 1].imag;
      itemp5 = arr[i + n / 2 + 1].imag;
    }
    if (i + 2 < n / 2) {
      rtemp3 = arr[i + 2].real;
      rtemp6 = arr[i + n / 2 + 2].real;
      itemp3 = arr[i + 2].imag;
      itemp6 = arr[i + n / 2 + 2].imag;
    }
    if (i + 3 < n / 2) {
      rtemp4 = arr[i + 3].real;
      rtemp7 = arr[i + n / 2 + 3].real;
      itemp4 = arr[i + 3].imag;
      itemp7 = arr[i + n / 2 + 3].imag;
    }

    ereal = _mm256_set_pd(rtemp4, rtemp3, rtemp2, rtemp1);
    eimag = _mm256_set_pd(itemp4, itemp3, itemp2, itemp1);
    oreal = _mm256_set_pd(rtemp7, rtemp6, rtemp5, arr[i + n / 2].real);
    oimag = _mm256_set_pd(itemp7, itemp6, itemp5, arr[i + n / 2].imag);
    wreal = _mm256_set_pd(cos(-2.0 * PI * (i + 3) / n), cos(-2.0 * PI * (i + 2) / n), cos(-2.0 * PI * (i + 1) / n), cos(-2.0 * PI * i / n));
    wimag = _mm256_set_pd(sin(-2.0 * PI * (i + 3) / n), sin(-2.0 * PI * (i + 2) / n), sin(-2.0 * PI * (i + 1) / n), sin(-2.0 * PI * i / n));
    w0real = _mm256_sub_pd(_mm256_mul_pd(wreal, oreal), _mm256_mul_pd(wimag, oimag)); 
    w0imag = _mm256_add_pd(_mm256_mul_pd(wreal, oimag), _mm256_mul_pd(wimag, oreal));
    freal = _mm256_add_pd(ereal, w0real);
    fimag = _mm256_add_pd(eimag, w0imag);
    sreal = _mm256_sub_pd(ereal, w0real);
    simag = _mm256_mul_pd(eimag, w0imag);
    
    arr[i].real = freal[0];
    arr[i].imag = fimag[0];
    arr[i + n / 2].real = sreal[0];
    arr[i + n / 2].imag = simag[0];

    if (i + 1 < n / 2) {
      arr[i + 1].real = freal[1];
      arr[i + 1].imag = fimag[1];
      arr[i + 1 + n / 2].real = sreal[1];
      arr[i + 1 + n / 2].imag = simag[1];
    }
    if (i + 2 < n / 2) {
      arr[i + 2].real = freal[2];
      arr[i + 2].imag = fimag[2];
      arr[i + 2 + n / 2].real = sreal[2];
      arr[i + 2 + n / 2].imag = simag[2];
    }
    if (i + 3 < n / 2) {
      arr[i + 3].real = freal[3];
      arr[i + 3].imag = fimag[3];
      arr[i + 3 + n / 2].real = sreal[3];
      arr[i + 3 + n / 2].imag = simag[3];
    }
  }
}

int main() {
  std::default_random_engine generator(42);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  const int NELEM = 8192;
  double* a = (double*)aligned_alloc(32, sizeof(double) * NELEM);
  double* b = (double*)aligned_alloc(32, sizeof(double) * NELEM);
  double* c = (double*)aligned_alloc(32, sizeof(double) * NELEM);
  Complex* ca = (Complex*)aligned_alloc(32, sizeof(Complex) * NELEM);
  Complex* cb = (Complex*)aligned_alloc(32, sizeof(Complex) * NELEM);
  Complex* cc = (Complex*)aligned_alloc(32, sizeof(Complex) * NELEM);

  for (int i = 0; i < NELEM; ++i) {
    a[i] = distribution(generator);
    b[i] = distribution(generator);
  }

  double st, et;

  // 1. Direct convolution
  st = get_time();
  for (int i = 0; i < NELEM; ++i) {
    c[i] = 0;
    for (int j = 0; j < NELEM; ++j) {
      c[i] += a[j] * b[(i - j + NELEM) % NELEM];
    }
  }
  et = get_time();
  printf("Direct convolution: %lf sec\n", et - st);

  // 2. Convolution with FFT
  st = get_time();
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
  et = get_time();
  printf("FFT convolution: %lf sec\n", et - st);

  // 3. Convolution with FFT (SIMD)
  st = get_time();
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
  et = get_time();
  printf("FFT (SIMD) convolution: %lf sec\n", et - st);

  // 4. Compare
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

  free(a); free(b); free(c);
  free(ca); free(cb); free(cc);

  return 0;
}