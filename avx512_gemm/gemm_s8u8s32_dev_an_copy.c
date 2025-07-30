#include <stdint.h>
#include <immintrin.h>

void gemm_s8u8s32_dev_an_copy(int m, int n, int lda, int8_t *p_a, int8_t *p_tmp)
{
  int i, j;

  __m128i xmm_0, xmm_1, xmm_2, xmm_3, xmm_4, xmm_5, xmm_6, xmm_7;

  for (i=0; i<m; i+=16) {
    for (j=0; j<n; j+=4) {
      xmm_0 = _mm_loadu_epi8(p_a+(j*lda)+i);
      xmm_1 = _mm_loadu_epi8(p_a+((j+1)*lda)+i);
      xmm_2 = _mm_loadu_epi8(p_a+((j+2)*lda)+i);
      xmm_3 = _mm_loadu_epi8(p_a+((j+3)*lda)+i);

      xmm_4 = _mm_unpacklo_epi8(xmm_0, xmm_2);
      xmm_5 = _mm_unpacklo_epi8(xmm_1, xmm_3);

      xmm_6 = _mm_unpacklo_epi8(xmm_4, xmm_5);
      xmm_7 = _mm_unpackhi_epi8(xmm_4, xmm_5);

      _mm_storeu_epi8(p_tmp, xmm_6);
      _mm_storeu_epi8(p_tmp+16, xmm_7);

      xmm_4 = _mm_unpackhi_epi8(xmm_0, xmm_2);
      xmm_5 = _mm_unpackhi_epi8(xmm_1, xmm_3);

      xmm_6 = _mm_unpacklo_epi8(xmm_4, xmm_5);
      xmm_7 = _mm_unpackhi_epi8(xmm_4, xmm_5);

      _mm_storeu_epi8(p_tmp+32, xmm_6);
      _mm_storeu_epi8(p_tmp+48, xmm_7);

      p_tmp += 64;
    }
  }
}
