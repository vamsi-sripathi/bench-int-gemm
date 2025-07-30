#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mkl.h"

#if defined BENCH_CBLAS
#include "mkl_cblas.h"
#if defined S8U8S32
#define A_TYPE      MKL_UINT8
#define B_TYPE      MKL_INT8
#define C_TYPE      MKL_INT32
#define FNAME       cblas_gemm_s8u8s32
#define FNAME_REF   gemm_s8u8s32_ref
#elif defined S8U8S32_EXPLICIT
#define A_TYPE      MKL_UINT8
#define B_TYPE      MKL_INT8
#define C_TYPE      MKL_INT32
#define FNAME       cblas_gemm_s8u8s32_explicit
#define FNAME_REF   gemm_s8u8s32_ref
#elif defined S8S8S32
#define A_TYPE      MKL_INT8
#define B_TYPE      MKL_INT8
#define C_TYPE      MKL_INT32
#define FNAME       cblas_gemm_s8s8s32
#define FNAME_REF   gemm_s8s8s32_ref
#endif
void FNAME(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
           const CBLAS_OFFSET offsetc, const MKL_INT m, const MKL_INT n, const MKL_INT k,
           const float alpha, const void *a, const MKL_INT lda, const MKL_INT8 oa,
           const void *b, const MKL_INT ldb, const MKL_INT8 ob, const float beta,
           MKL_INT32 *c, const MKL_INT ldc, const MKL_INT32 *oc);
#else
#if defined S8U8S32
#define A_TYPE      MKL_INT8
#define B_TYPE      MKL_UINT8
#define C_TYPE      MKL_INT32
#define FNAME       gemm_s8u8s32_dev
#define FNAME_REF   gemm_s8u8s32_ref
#elif S8S8S32
#define A_TYPE      MKL_INT8
#define B_TYPE      MKL_INT8
#define C_TYPE      MKL_INT32
#define FNAME       gemm_s8s8s32
#define FNAME_REF   gemm_s8s8s32_ref
#elif defined S8U8S32_EXPLICIT
#define A_TYPE      MKL_INT8
#define B_TYPE      MKL_UINT8
#define C_TYPE      MKL_INT32
#define FNAME       gemm_s8u8s32_explicit
#define FNAME_REF   gemm_s8u8s32_ref
#elif defined S8U8S32_PATCHED
#define A_TYPE      MKL_INT8
#define B_TYPE      MKL_UINT8
#define C_TYPE      MKL_INT32
#define FNAME       gemm_s8u8s32_patched
#define FNAME_REF   gemm_s8u8s32_ref
#else
#error undefined INT_TYPE
#endif
#endif

#define COMPSIZE 1

#define STRINGIFY(x)  STRINGIFY_(x)
#define STRINGIFY_(x) #x

#define  NUM_TRIALS       (30)
#define  PAGE_ALIGNMENT   (4096)
#define  AVX512_ALIGNMENT (64)

/* #define  VERBOSE */
/* #define  PAD_LD */
/* #define  FORCE_UNALIGN */

static inline int check_result(int m, int n, int ld, C_TYPE *c, C_TYPE *c_ref)
{
  int i, j;

#ifdef PRINT_MATRIX
  printf ("Matrix-C w/ MKL\n");
  for (j=0; j<m; j++) {
    for (i=0; i<n; i++) {
#ifdef BENCH_CBLAS
      printf (" %d ", c[j*ld+i]);
#else
      printf (" %d ", c[i*ld+j]);
#endif
    }
    printf ("\n");
  }

  printf ("Matrix-C w/ Ref\n");
  for (j=0; j<m; j++) {
    for (i=0; i<n; i++) {
#ifdef BENCH_CBLAS
      printf (" %d ", c_ref[j*ld+i]);
#else
      printf (" %d ", c_ref[i*ld+j]);
#endif
    }
    printf ("\n");
  }
#endif

  for (j=0; j<n; j++) {
    for (i=0; i<m; i++) {
#ifdef BENCH_CBLAS
      if (c[i*ld+j] != c_ref[i*ld+j]) {
        printf ("\n ERROR: [%d][%d] ref = %d, mkl = %d\n",i,j,c_ref[i*ld+j], c[i*ld+j]);
        return 1;
      }
#else
      if (c[j*ld+i] != c_ref[j*ld+i]) {
        printf ("\n ERROR: [%d][%d] ref = %d, mkl = %d\n",i,j,c_ref[j*ld+i], c[j*ld+i]);
        return 1;
      }
#endif
    }
  }
  return 0;
}

static inline int fix_ld(int ld)
{
  int bad_ld_alignment = 256*sizeof(MKL_INT8);
  int cache_line_alignment = 64;
  int ld_bytes = ld*sizeof(MKL_INT8);
  int padded_ld;

#if 0
  return ((ld_bytes + bad_ld_alignment - 1)/bad_ld_alignment*bad_ld_alignment + cache_line_alignment)/sizeof(MKL_INT8);
#endif

  if (ld_bytes%bad_ld_alignment) {
    padded_ld = ((ld_bytes + cache_line_alignment - 1)/cache_line_alignment)*cache_line_alignment;
    if (padded_ld%bad_ld_alignment) {
      return padded_ld/sizeof(MKL_INT8);
    } else {
      return (padded_ld+cache_line_alignment)/sizeof(MKL_INT8);
    }
  } else {
    return (ld_bytes + cache_line_alignment)/sizeof(MKL_INT8);
  }
}

int main (int argc, char **argv)
{
  int s = mkl_enable_instructions(MKL_ENABLE_AVX512_E1);
  char       transa, transb, off_type;
  int        m, n, k, lda, ldb, ldc;
  int        i, j, t, ax, bx, cx;
  int        a_rows, a_cols, b_rows, b_cols, c_rows, c_cols;
  double     t_start, t_stop, curr_gemm_gflops, best_gemm_gflops;
  double     curr_gemm_time, best_gemm_time;
  int        mem_offset = 0;
  float      alpha = 1.0, beta = 0.0;
  A_TYPE     *a = NULL;
  B_TYPE     *b = NULL;
  C_TYPE     *c = NULL;

  MKL_INT32 zero_offset = 0;
  MKL_INT8 off_a, off_b;
  MKL_INT32 *off_c;

  if (argc != 9) {
    printf ("\n USAGE: %s transa<n|t> transb<n|t> off_type<r|c|f> m<int> n<int> k<int> off_a off_b\n", argv[0]); fflush(0);
    exit (1);
  }

  transa      = argv[1][0];
  transb      = argv[2][0];
  off_type    = argv[3][0];
  m           = atoi(argv[4]);
  n           = atoi(argv[5]);
  k           = atoi(argv[6]);
  off_a       = atoi(argv[7]);
  off_b       = atoi(argv[8]);

#ifdef BENCH_CBLAS
  ax     = ((transa == 'N' || transa == 'n') ? m : k);
  bx     = ((transb == 'N' || transb == 'n') ? k : n);
  lda    = ((transa == 'N' || transa == 'n') ? k : m);
  ldb    = ((transb == 'N' || transb == 'n') ? n : k);
  ldc    = n;
  a_rows = ax;
  a_cols = lda;
  b_rows = bx;
  b_cols = ldb;
  c_rows = m;
  c_cols = ldc;
  cx     = m;

  CBLAS_LAYOUT layout = CblasRowMajor;
  CBLAS_TRANSPOSE cblas_transa = ((transa == 'N' || transa == 'n') ? CblasNoTrans : CblasTrans);
  CBLAS_TRANSPOSE cblas_transb = ((transb == 'N' || transb == 'n') ? CblasNoTrans : CblasTrans);
  CBLAS_OFFSET cblas_offtype;
  if (off_type == 'f' || off_type == 'F') {
    cblas_offtype = CblasFixOffset;
  } else {
    printf ("\nERROR: CBLAS tester currently does not support non-fixed offset type for C\n");
    exit(1);
  }
#else
  ax     = ((transa == 'N' || transa == 'n') ? k : m);
  bx     = ((transb == 'N' || transb == 'n') ? n : k);
  lda    = ((transa == 'N' || transa == 'n') ? m : k);
  ldb    = ((transb == 'N' || transb == 'n') ? k : n);
  ldc    = m;
  a_rows = lda;
  a_cols = ax;
  b_rows = ldb;
  b_cols = bx;
  c_rows = ldc;
  c_cols = n;
  cx     = n;
#endif

#ifdef PAD_LD
  lda = fix_ld(lda);
  ldb = fix_ld(ldb);
  ldc = fix_ld(ldc);
#endif

#ifdef FORCE_UNALIGN
  mem_offset++;
#endif

  A_TYPE *a_mem = (A_TYPE *) mkl_malloc (((lda*ax)+mem_offset)*sizeof(A_TYPE), AVX512_ALIGNMENT);
  B_TYPE *b_mem = (B_TYPE *) mkl_malloc (((ldb*bx)+mem_offset)*sizeof(B_TYPE), AVX512_ALIGNMENT);
  C_TYPE *c_mem = (C_TYPE *) mkl_malloc (((ldc*cx)+mem_offset)*sizeof(C_TYPE),  AVX512_ALIGNMENT);
#ifdef VALIDATION
  C_TYPE *c_ref = (C_TYPE *) mkl_malloc (((ldc*cx)+mem_offset)*sizeof(C_TYPE),  AVX512_ALIGNMENT);
#endif

  if (off_type == 'f' || off_type == 'F') {
    off_c = &zero_offset;
  } else if (off_type == 'r' || off_type == 'R') {
    C_TYPE *off_c = (C_TYPE *) mkl_malloc (n*sizeof(C_TYPE), AVX512_ALIGNMENT); 
    for (i=0; i<n; i++) {
      off_c[i] = rand()%20;
    }
  } else if (off_type == 'c' || off_type == 'C') {
    C_TYPE *off_c = (C_TYPE *) mkl_malloc (m*sizeof(C_TYPE), AVX512_ALIGNMENT); 
    for (i=0; i<m; i++) {
      off_c[i] = rand()%20;
    }
  }

  a = a_mem+mem_offset;
  b = b_mem+mem_offset;
  c = c_mem+mem_offset;

#ifdef FORCE_UNALIGN
  printf ("Alignment offset of A from %s byte boundary = %ld\n",
          STRINGIFY(AVX512_ALIGNMENT), (unsigned long)a%AVX512_ALIGNMENT);fflush(0);
  printf ("Alignment offset of B from %s byte boundary = %ld\n",
          STRINGIFY(AVX512_ALIGNMENT), (unsigned long)b%AVX512_ALIGNMENT);fflush(0);
  printf ("Alignment offset of C from %s byte boundary = %ld\n",
          STRINGIFY(AVX512_ALIGNMENT), (unsigned long)c%AVX512_ALIGNMENT);fflush(0);
#endif

  for (j=0; j<a_cols; j++) {
    for (i=0; i<a_rows; i++) {
#ifdef BENCH_CBLAS
#if defined (S8U8S32) || defined (S8U8S32_EXPLICIT)
#ifdef FILL_MATRIX
      a[(i*lda)+j] = 128 + rand()%127;
#else
      a[(i*lda)+j] = rand()%127;
#endif
#elif S8S8S32
      a[(i*lda)+j] = 64 + rand()%63 * ((i%2) ? -1 : 1);
#endif
#else
#ifdef FILL_MATRIX
      a[(j*lda)+i] = 64 + rand()%63;
#else
      a[(j*lda)+i] = rand()%63;
#endif
#endif
    }
  }

#ifdef PRINT_MATRIX
  printf ("Matrix-A\n");
  for (j=0; j<a_rows; j++) {
    for (i=0; i<a_cols; i++) {
#ifdef BENCH_CBLAS
      printf (" %d ",a[j*lda+i]);
#else
      printf (" %d ",a[i*lda+j]);
#endif
    } 
    printf ("\n");
  }
#endif

  for (j=0; j<b_cols; j++) {
    for (i=0; i<b_rows; i++) {
#ifdef BENCH_CBLAS
#if defined (S8U8S32) || defined (S8S8S32) || defined (S8U8S32_EXPLICIT)
#ifdef FILL_MATRIX
      b[(i*ldb)+j] = 64 + rand()%63 * ((i%2) ? -1 : 1);
#else
      b[(i*ldb)+j] = rand()%63;
#endif
#endif
#else
#ifdef S8S8S32
      b[(j*ldb)+i] = rand()%127 * ((i%2) ? -1 : 1);
#else
#ifdef FILL_MATRIX
      b[(j*ldb)+i] = 128 + rand()%127;
#else
      b[(j*ldb)+i] = rand()%127;
#endif
#endif
#endif
    }
  }

#ifdef PRINT_MATRIX
  printf ("Matrix-B\n");
  for (j=0; j<b_rows; j++) {
    for (i=0; i<b_cols; i++) {
#ifdef BENCH_CBLAS
      printf (" %d ",b[j*ldb+i]);
#else
      printf (" %d ",b[i*ldb+j]);
#endif
    } 
    printf ("\n");
  }
#endif

#if 0
  for (j=0; j<c_cols; j++) {
    for (i=0; i<c_rows; i++) {
      c[(j*ldc)+i] = 0;
#ifdef VALIDATION
      c_ref[(j*ldc)+i] = 0;
#endif
    }
  }
#endif

  // Warm-up call
  t_start = dsecnd();
  t_start = dsecnd();
#ifdef BENCH_CBLAS
  FNAME (layout, cblas_transa, cblas_transb, cblas_offtype, m, n, k, alpha, a, lda, off_a, b, ldb, off_b, beta, c, ldc, off_c);
#else
  FNAME (&transa, &transb, &off_type, &m ,&n, &k, &alpha, a, &lda, &off_a, b, &ldb, &off_b, &beta, c, &ldc, off_c);
#endif

#ifdef VALIDATION
#ifdef BENCH_CBLAS
  FNAME_REF(&transb, &transa, &off_type, &n ,&m, &k, &alpha, b, &ldb, &off_b, a, &lda, &off_a, &beta, c_ref, &ldc, off_c);
  /* FNAME_REF(layout, cblas_transa, cblas_transb, cblas_offtype, m, n, k, alpha, a, lda, off_a, b, ldb, off_b, beta, c_ref, ldc, off_c); */
#else
  FNAME_REF(&transa, &transb, &off_type, &m ,&n, &k, &alpha, a, &lda, &off_a, b, &ldb, &off_b, &beta, c_ref, &ldc, off_c);
#endif
  if (check_result(m, n, ldc, c, c_ref)) {
    printf ("%s: transa= %c, transb= %c, off_type= %c, m= %d, n= %d, k= %d, lda= %d, ldb= %d, ldc= %d, off_a= %d, off_b= %d, VALIDATION FAILED\n",
            STRINGIFY(FNAME), transa, transb, off_type, m, n, k, lda, ldb, ldc, off_a, off_b); fflush(0);
    exit(1);
  } else {
    printf ("%s: transa= %c, transb= %c, off_type= %c, m= %d, n= %d, k= %d, lda= %d, ldb= %d, ldc= %d, off_a= %d, off_b= %d, VALIDATION PASSED\n",
            STRINGIFY(FNAME), transa, transb, off_type, m, n, k, lda, ldb, ldc, off_a, off_b); fflush(0);
  }
#endif

  long long n_ops = 2LL * COMPSIZE * m * n * k;
  double avg_gemm_gflops = 0.0;
  double avg_gemm_time = 0.0;

  for (t=0; t<NUM_TRIALS; t++) {
    t_start = dsecnd();
#ifdef BENCH_CBLAS
  FNAME (layout, cblas_transa, cblas_transb, cblas_offtype, m, n, k, alpha, a, lda, off_a, b, ldb, off_b, beta, c, ldc, off_c);
#else
  FNAME (&transa, &transb, &off_type, &m ,&n, &k, &alpha, a, &lda, &off_a, b, &ldb, &off_b, &beta, c, &ldc, off_c);
#endif
    t_stop = dsecnd();

    curr_gemm_gflops = n_ops/(t_stop-t_start) * 1.e-9;
    avg_gemm_gflops += curr_gemm_gflops;
    curr_gemm_time  = t_stop-t_start;
    avg_gemm_time += curr_gemm_time;

    if (t) {
      if (curr_gemm_gflops > best_gemm_gflops) {
        best_gemm_gflops = curr_gemm_gflops;
      }
      if (curr_gemm_time < best_gemm_time) {
        best_gemm_time = curr_gemm_time;
      }
    } else {
      best_gemm_gflops = curr_gemm_gflops;
      best_gemm_time   = curr_gemm_time;
    }
#ifdef VERBOSE
    printf ("%s: transa= %c, transb= %c, off_type= %c, m= %d, n= %d, k= %d, lda= %d, ldb= %d, ldc= %d, off_a= %d, off_b= %d, gflops= %.2f, time(ms)= %.2f\n",
            STRINGIFY(FNAME), transa, transb, off_type, m, n, k, lda, ldb, ldc, off_a, off_b,
            curr_gemm_gflops, curr_gemm_time*1.e3); fflush(0);
#endif
  }
  printf ("%s: transa= %c, transb= %c, off_type= %c, m= %d, n= %d, k= %d, lda= %d, ldb= %d, ldc= %d, off_a= %d, off_b= %d, avg-gflops= %.2f, best-gflops= %.2f, avg-time(ms)= %.2f, best-time(ms)= %.2f\n",
          STRINGIFY(FNAME), transa, transb, off_type, m, n, k, lda, ldb, ldc, off_a, off_b,
          avg_gemm_gflops/NUM_TRIALS, best_gemm_gflops, avg_gemm_time*1.e3, best_gemm_time*1.e3); fflush(0);

  mkl_free (a_mem);
  mkl_free (b_mem);
  mkl_free (c_mem);

  return 0;
}
