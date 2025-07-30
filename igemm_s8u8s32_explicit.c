#include "mkl.h"
#include "omp.h"

#define  FNAME   gemm_s8u8s32_explicit
#define  A_TYPE  MKL_INT8
#define  B_TYPE  MKL_UINT8
#define  C_TYPE  MKL_INT32

static inline void matrix_add(char *off_type, int *p_m, int *p_n, int *p_ld, MKL_INT32 *c, MKL_INT32 *oc)
{
  int i, j;
  int m, n, ld;

  m = *p_m;
  n = *p_n;
  ld = *p_ld;

  if (off_type == "f" || off_type == "F") {
#pragma omp parallel for default(shared) private(i,j)
    for (i=0; i<n; i++) {
#pragma unroll
      for (j=0; j<m; j++) {
        c[i*ld+j] += *oc;
      }
    }
  } else if (off_type == "c" || off_type == "C") {
#pragma omp parallel for default(shared) private(i,j)
    for (i=0; i<n; i++) {
#pragma unroll
      for (j=0; j<m; j++) {
        c[i*ld+j] += oc[j];
      }
    }
  } else if (off_type == "r" || off_type == "R") {
#pragma omp parallel for default(shared) private(i,j)
    for (i=0; i<n; i++) {
#pragma unroll
      for (j=0; j<m; j++) {
        c[i*ld+j] += oc[i];
      }
    }
  }
}

void FNAME (char *p_transa, char *p_transb, char *p_off_type, int *p_m, int *p_n, int *p_k,
            float *p_alpha, A_TYPE *p_a, int *p_lda, MKL_INT8 *p_oa,
            B_TYPE *p_b, int *p_ldb, MKL_INT8 *p_ob,
            float *p_beta, C_TYPE *p_c, int *p_ldc, MKL_INT32 *p_oc)
{
  char transa, transb;
  int m, n, k, lda, ldb, ldc;
  MKL_INT8 offset_a, offset_b;
  int transpose_a, transpose_b;
  int i, j;
  MKL_INT32 tmp;
  MKL_INT32 *a_oc = NULL, *b_oc = NULL;

  transa = *p_transa;
  transb = *p_transb;
  m = *p_m;
  n = *p_n;
  k = *p_k;
  lda = *p_lda;
  ldb = *p_ldb;
  ldc = *p_ldc;
  transpose_a = ((transa == 'N' || transa == 'n') ? 0 : 1);
  transpose_b = ((transb == 'N' || transb == 'n') ? 0 : 1);

  offset_a = *p_oa;
  offset_b = *p_ob;

  MKL_INT8 zero_oa = 0, zero_ob = 0;
  MKL_INT32 zero_oc = 0;

  if (offset_a) {
    b_oc = (MKL_INT32 *)mkl_malloc(sizeof(MKL_INT32)*n, 64);

    // find sum of column of op(B)
    if (transpose_b) {
#pragma omp parallel for default(shared) private(tmp,i,j)
      for (i=0; i<n; i++) {
        tmp = 0;
#pragma unroll
        for (j=0; j<k; j++) {
          tmp += p_b[j*ldb + i];
        }
        tmp *= offset_a;
        /* tmp += k*offset_b*offset_a; */
        b_oc[i] = tmp;
      }
    } else {
#pragma omp parallel for default(shared) private(tmp,i,j)
      for (i=0; i<n; i++) {
        tmp = 0;
#pragma unroll
        for (j=0; j<k; j++) {
          tmp += p_b[i*ldb + j];
        }
        tmp *= offset_a;
        /* tmp += k*offset_b*offset_a; */
        b_oc[i] = tmp;
      }
    }
  }

  if (offset_b) {
    a_oc = (MKL_INT32 *)mkl_malloc(sizeof(MKL_INT32)*m, 64);
    // find sum of row of op(A)
    if (transpose_a) {
#pragma omp parallel for default(shared) private(tmp,i,j)
      for (i=0; i<m; i++) {
        tmp = 0;
#pragma unroll
        for (j=0; j<k; j++) {
          tmp += p_a[i*lda + j];
        }
        tmp *= offset_b;
        /* tmp += k*offset_b*offset_a; */
        a_oc[i] = tmp;
      }
    } else {
#pragma omp parallel for default(shared) private(tmp,i,j)
      for (i=0; i<m; i++) {
        tmp = 0;
#pragma unroll
        for (j=0; j<k; j++) {
          tmp += p_a[j*lda + i];
        }
        tmp *= offset_b;
        /* tmp += k*offset_b*offset_a; */
        a_oc[i] = tmp;
      }
    }
  }

  // Always call with fixed offset and zero
  gemm_s8u8s32(p_transa, p_transb, "f", p_m, p_n, p_k, p_alpha, p_a, p_lda, &zero_oa, p_b, p_ldb, &zero_ob, p_beta, p_c, p_ldc, &zero_oc);

  if ((!offset_a) && (offset_b)) {
    matrix_add("c", p_m, p_n, p_ldc, p_c, a_oc);
  } else if ((offset_a) && (!offset_b)) {
    matrix_add("r", p_m, p_n, p_ldc, p_c, b_oc);
  } else if ((offset_a) && (offset_b)) {
#pragma omp parallel for default(shared) private(i,j)
    for (j=0; j<n; j++) {
#pragma unroll
      for (i=0; i<m; i++) {
        p_c[j*ldc+i] += b_oc[j] + a_oc[i] + (k*offset_a*offset_b);
      }
    }
  }

  if (a_oc) {
    mkl_free(a_oc);
  }
  if (b_oc) {
    mkl_free(b_oc);
  }
}
