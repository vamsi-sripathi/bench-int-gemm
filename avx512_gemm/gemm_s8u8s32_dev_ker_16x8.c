#include <stdint.h>
#include <immintrin.h>

void gemm_s8u8s32_dev_ker_16x8(int m, int n, int k, int8_t *p_a_tmp, uint8_t *p_b_tmp, int32_t *p_c, int ldc, int beta)
{
  int i, j, l;
  __m512i zmm_a0, zmm_b0, zmm_t0, zmm_c0, zmm_c1, zmm_c2, zmm_c3, zmm_one;
  __m512i zmm_c4, zmm_c5, zmm_c6, zmm_c7;
  zmm_one = _mm512_set1_epi16(1);

  for (i=0; i<m; i+=16) {
    for (j=0; j<n; j+=8) {
      if (beta == 0) {
       zmm_c0 = _mm512_setzero_epi32();
       zmm_c1 = _mm512_setzero_epi32();
       zmm_c2 = _mm512_setzero_epi32();
       zmm_c3 = _mm512_setzero_epi32();
       zmm_c4 = _mm512_setzero_epi32();
       zmm_c5 = _mm512_setzero_epi32();
       zmm_c6 = _mm512_setzero_epi32();
       zmm_c7 = _mm512_setzero_epi32();
      } else {
       zmm_c0 = _mm512_loadu_epi32(&p_c[j*ldc+i]);
       zmm_c1 = _mm512_loadu_epi32(&p_c[(j+1)*ldc+i]);
       zmm_c2 = _mm512_loadu_epi32(&p_c[(j+2)*ldc+i]);
       zmm_c3 = _mm512_loadu_epi32(&p_c[(j+3)*ldc+i]);
       zmm_c4 = _mm512_loadu_epi32(&p_c[(j+4)*ldc+i]);
       zmm_c5 = _mm512_loadu_epi32(&p_c[(j+5)*ldc+i]);
       zmm_c6 = _mm512_loadu_epi32(&p_c[(j+6)*ldc+i]);
       zmm_c7 = _mm512_loadu_epi32(&p_c[(j+7)*ldc+i]);
      }
       for (l=0; l<k; l+=16, p_a_tmp+=256, p_b_tmp+=128) {
         zmm_a0 = _mm512_loadu_epi8(p_a_tmp);
         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c0 = _mm512_add_epi32(zmm_c0, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+4));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c1 = _mm512_add_epi32(zmm_c1, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+8));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c2 = _mm512_add_epi32(zmm_c2, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+12));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c3 = _mm512_add_epi32(zmm_c3, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+16));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c4 = _mm512_add_epi32(zmm_c4, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+20));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c5 = _mm512_add_epi32(zmm_c5, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+24));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c6 = _mm512_add_epi32(zmm_c6, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+28));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c7 = _mm512_add_epi32(zmm_c7, zmm_t0);







         zmm_a0 = _mm512_loadu_epi8(p_a_tmp+64);
         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+32));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c0 = _mm512_add_epi32(zmm_c0, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+36));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c1 = _mm512_add_epi32(zmm_c1, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+40));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c2 = _mm512_add_epi32(zmm_c2, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+44));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c3 = _mm512_add_epi32(zmm_c3, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+48));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c4 = _mm512_add_epi32(zmm_c4, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+52));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c5 = _mm512_add_epi32(zmm_c5, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+56));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c6 = _mm512_add_epi32(zmm_c6, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+60));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c7 = _mm512_add_epi32(zmm_c7, zmm_t0);




         zmm_a0 = _mm512_loadu_epi8(p_a_tmp+128);
         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+64));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c0 = _mm512_add_epi32(zmm_c0, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+68));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c1 = _mm512_add_epi32(zmm_c1, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+72));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c2 = _mm512_add_epi32(zmm_c2, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+76));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c3 = _mm512_add_epi32(zmm_c3, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+80));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c4 = _mm512_add_epi32(zmm_c4, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+84));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c5 = _mm512_add_epi32(zmm_c5, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+88));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c6 = _mm512_add_epi32(zmm_c6, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+92));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c7 = _mm512_add_epi32(zmm_c7, zmm_t0);




         zmm_a0 = _mm512_loadu_epi8(p_a_tmp+192);
         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+96));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c0 = _mm512_add_epi32(zmm_c0, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+100));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c1 = _mm512_add_epi32(zmm_c1, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+104));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c2 = _mm512_add_epi32(zmm_c2, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+108));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c3 = _mm512_add_epi32(zmm_c3, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+112));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c4 = _mm512_add_epi32(zmm_c4, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+116));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c5 = _mm512_add_epi32(zmm_c5, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+120));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c6 = _mm512_add_epi32(zmm_c6, zmm_t0);

         zmm_b0 = _mm512_broadcastd_epi32(_mm_loadu_si32(p_b_tmp+124));
         zmm_t0 = _mm512_maddubs_epi16(zmm_b0, zmm_a0);
         zmm_t0 = _mm512_madd_epi16(zmm_t0, zmm_one);
         zmm_c7 = _mm512_add_epi32(zmm_c7, zmm_t0);

       }
       _mm512_storeu_epi32(&p_c[j*ldc+i], zmm_c0);
       _mm512_storeu_epi32(&p_c[(j+1)*ldc+i], zmm_c1);
       _mm512_storeu_epi32(&p_c[(j+2)*ldc+i], zmm_c2);
       _mm512_storeu_epi32(&p_c[(j+3)*ldc+i], zmm_c3);
       _mm512_storeu_epi32(&p_c[(j+4)*ldc+i], zmm_c4);
       _mm512_storeu_epi32(&p_c[(j+5)*ldc+i], zmm_c5);
       _mm512_storeu_epi32(&p_c[(j+6)*ldc+i], zmm_c6);
       _mm512_storeu_epi32(&p_c[(j+7)*ldc+i], zmm_c7);
    }
  }
}
