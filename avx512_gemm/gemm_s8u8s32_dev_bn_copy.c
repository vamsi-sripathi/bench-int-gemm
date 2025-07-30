#include <stdint.h>
#include <immintrin.h>

#define N_UNROLL  (8)

void gemm_s8u8s32_dev_bn_copy(int m, int n, int ldb, uint8_t *p_b, uint8_t *p_tmp)
{
  int i, j;

  __m128i xmm_0, xmm_1, xmm_2, xmm_3, xmm_4, xmm_5, xmm_6, xmm_7, xmm_8, xmm_9;

  for (i=0; i<m; i+=16) {
    for (j=0; j<n; j+=N_UNROLL) {
#if N_UNROLL == 8
      xmm_0 = _mm_loadu_epi8(p_b+(j*ldb)+i);
      xmm_1 = _mm_loadu_epi8(p_b+((j+1)*ldb)+i);
      xmm_2 = _mm_loadu_epi8(p_b+((j+2)*ldb)+i);
      xmm_3 = _mm_loadu_epi8(p_b+((j+3)*ldb)+i);

      xmm_4 = _mm_unpacklo_epi32(xmm_0, xmm_2);
      xmm_5 = _mm_unpacklo_epi32(xmm_1, xmm_3);

      xmm_6 = _mm_unpacklo_epi32(xmm_4, xmm_5);
      xmm_7 = _mm_unpackhi_epi32(xmm_4, xmm_5);

      _mm_storeu_epi8(p_tmp, xmm_6);
      _mm_storeu_epi8(p_tmp+32, xmm_7);

      xmm_4 = _mm_unpackhi_epi32(xmm_0, xmm_2);
      xmm_5 = _mm_unpackhi_epi32(xmm_1, xmm_3);

      xmm_6 = _mm_unpacklo_epi32(xmm_4, xmm_5);
      xmm_7 = _mm_unpackhi_epi32(xmm_4, xmm_5);

      _mm_storeu_epi8(p_tmp+64, xmm_6);
      _mm_storeu_epi8(p_tmp+96, xmm_7);

      xmm_0 = _mm_loadu_epi8(p_b+((j+4)*ldb)+i);
      xmm_1 = _mm_loadu_epi8(p_b+((j+5)*ldb)+i);
      xmm_2 = _mm_loadu_epi8(p_b+((j+6)*ldb)+i);
      xmm_3 = _mm_loadu_epi8(p_b+((j+7)*ldb)+i);

      xmm_4 = _mm_unpacklo_epi32(xmm_0, xmm_2);
      xmm_5 = _mm_unpacklo_epi32(xmm_1, xmm_3);

      xmm_8 = _mm_unpacklo_epi32(xmm_4, xmm_5);
      xmm_9 = _mm_unpackhi_epi32(xmm_4, xmm_5);

      _mm_storeu_epi8(p_tmp+16, xmm_8);
      _mm_storeu_epi8(p_tmp+48, xmm_9);

      xmm_4 = _mm_unpackhi_epi32(xmm_0, xmm_2);
      xmm_5 = _mm_unpackhi_epi32(xmm_1, xmm_3);

      xmm_8 = _mm_unpacklo_epi32(xmm_4, xmm_5);
      xmm_9 = _mm_unpackhi_epi32(xmm_4, xmm_5);

      _mm_storeu_epi8(p_tmp+80, xmm_8);
      _mm_storeu_epi8(p_tmp+112, xmm_9);


      p_tmp += 128;
#else
      xmm_0 = _mm_loadu_epi8(p_b+(j*ldb)+i);
      xmm_1 = _mm_loadu_epi8(p_b+((j+1)*ldb)+i);
      xmm_2 = _mm_loadu_epi8(p_b+((j+2)*ldb)+i);
      xmm_3 = _mm_loadu_epi8(p_b+((j+3)*ldb)+i);

      xmm_4 = _mm_unpacklo_epi32(xmm_0, xmm_2);
      xmm_5 = _mm_unpacklo_epi32(xmm_1, xmm_3);

      xmm_6 = _mm_unpacklo_epi32(xmm_4, xmm_5);
      xmm_7 = _mm_unpackhi_epi32(xmm_4, xmm_5);

      _mm_storeu_epi8(p_tmp, xmm_6);
      _mm_storeu_epi8(p_tmp+16, xmm_7);

      xmm_4 = _mm_unpackhi_epi32(xmm_0, xmm_2);
      xmm_5 = _mm_unpackhi_epi32(xmm_1, xmm_3);

      xmm_6 = _mm_unpacklo_epi32(xmm_4, xmm_5);
      xmm_7 = _mm_unpackhi_epi32(xmm_4, xmm_5);

      _mm_storeu_epi8(p_tmp+32, xmm_6);
      _mm_storeu_epi8(p_tmp+48, xmm_7);

      p_tmp += 64;
#endif
    }
  }
}
