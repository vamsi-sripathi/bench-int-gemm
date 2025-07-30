#include "mkl.h"
#include "mkl_blas.h"
#include "omp.h"

#define  FNAME   gemm_s8s8s32
#define  A_TYPE  MKL_INT8
#define  B_TYPE  MKL_INT8
#define  C_TYPE  MKL_INT32

void FNAME(char *p_transa, char *p_transb, char *p_off_type, int *p_m, int *p_n, int *p_k,
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

  MKL_UINT8 *p_b_tmp = (MKL_UINT8 *)mkl_malloc(sizeof(MKL_UINT8)*k*n, 64);
  int ldb_tmp = ((transpose_b) ? n : k);
  MKL_INT8 offset_b_tmp = offset_b + -128;

  if (transpose_b) {
#pragma omp parallel for default(shared) private(i,j)
    for (i=0; i<k; i++) {
#pragma unroll
      for (j=0; j<n; j++) {
        p_b_tmp[i*ldb_tmp+j] = p_b[i*ldb+j] + 128;
      }
    }
  } else {
#pragma omp parallel for default(shared) private(i,j)
    for(i=0; i<n; i++) {
#pragma unroll
      for (j=0; j<k; j++) {
        p_b_tmp[i*ldb_tmp+j] = p_b[i*ldb+j] + 128;
      }
    }
  }

  gemm_s8u8s32_explicit(p_transa, p_transb, p_off_type, p_m, p_n, p_k, p_alpha, p_a, p_lda, p_oa, p_b_tmp, &ldb_tmp, &offset_b_tmp, p_beta, p_c, p_ldc, p_oc);

  mkl_free(p_b_tmp);
}
