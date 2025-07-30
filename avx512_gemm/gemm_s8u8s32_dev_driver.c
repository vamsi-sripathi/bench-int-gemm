#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>

void gemm_s8u8s32_dev_an_copy(int, int, int, int8_t *, int8_t *);
void gemm_s8u8s32_dev_bn_copy(int, int, int, uint8_t *, uint8_t *);
void gemm_s8u8s32_dev_ker_16x4(int, int, int, int8_t *, uint8_t *, int32_t *, int, int);
void gemm_s8u8s32_dev_ker_16x8(int, int, int, int8_t *, uint8_t *, int32_t *, int, int);

#define  M_BLOCK  (16)
#define  N_BLOCK  (8)
#define  K_BLOCK  (256)
#define  K_UNROLL (16)

void gemm_s8u8s32_dev(char *p_transa, char *p_transb, char *p_off_type, int *p_m, int *p_n, int *p_k,
                      float *p_alpha, int8_t *p_a, int *p_lda, int8_t *p_oa,
                      uint8_t *p_b, int *p_ldb, int8_t *p_ob,
                      float *p_beta, int32_t *p_c, int *p_ldc, int32_t *p_oc)
{
  char transa, transb;
  int m, n, k, lda, ldb, ldc;
  int8_t offset_a, offset_b;
  int transpose_a, transpose_b;
  int i, j, l, ll, size_k;

  transa = *p_transa;
  transb = *p_transb;
  transpose_a = ((transa == 'N' || transa == 'n') ? 0 : 1);
  transpose_b = ((transb == 'N' || transb == 'n') ? 0 : 1);
  offset_a = *p_oa;
  offset_b = *p_ob;
  m = *p_m;
  n = *p_n;
  k = *p_k;
  lda = *p_lda;
  ldb = *p_ldb;
  ldc = *p_ldc;

  if (m%M_BLOCK != 0 ||
      n%N_BLOCK  != 0 ||
      k%K_UNROLL != 0) {
    printf("config not supported\n");
    exit(1);
  }

  int m_blk = ((m > M_BLOCK) ? M_BLOCK : m);
  int n_blk = ((n > N_BLOCK) ? N_BLOCK : n);
  int k_blk = ((k > K_BLOCK) ? K_BLOCK : k);
  int k_unroll = K_UNROLL;

  int8_t *p_a_tmp = (int8_t *)_mm_malloc(sizeof(int8_t)*m_blk*k_blk, 64);
  uint8_t *p_b_tmp = (uint8_t *)_mm_malloc(sizeof(uint8_t)*k_blk*n_blk, 64);


  for (i=0; i<m; i+=m_blk) {
    for (j=0; j<k; j+=size_k) {
      if (j+k_blk <= k) {
        size_k = k_blk;
      } else {
        size_k = k_unroll;
      }
      gemm_s8u8s32_dev_an_copy(m_blk, size_k, lda, p_a+(j*lda)+i, p_a_tmp);
      for (l=0; l<n; l+=n_blk) {
        gemm_s8u8s32_dev_bn_copy(size_k, n_blk, ldb, p_b+(l*ldb)+j, p_b_tmp);
        if (j>0) {
          gemm_s8u8s32_dev_ker_16x8(m_blk, n_blk, size_k, p_a_tmp, p_b_tmp, p_c+(l*ldc)+i, ldc, 1);
        } else {
          gemm_s8u8s32_dev_ker_16x8(m_blk, n_blk, size_k, p_a_tmp, p_b_tmp, p_c+(l*ldc)+i, ldc, 0);
        }
      }
    }
  }

  _mm_free(p_a_tmp);
  _mm_free(p_b_tmp);
}
