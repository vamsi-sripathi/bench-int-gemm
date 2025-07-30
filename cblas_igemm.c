#include "mkl_cblas.h"
#include "mkl_blas.h"

#ifdef LIB_BUILD
#if defined S8U8S32
#define  CBLAS_FNAME   cblas_gemm_s8u8s32
#define  FNAME         gemm_s8u8s32_explicit
#elif defined S8S8S32
#define  CBLAS_FNAME   cblas_gemm_s8s8s32
#define  FNAME         gemm_s8s8s32
#else
#error "undefined INT type"
#endif
#else
#if defined S8U8S32
#define  CBLAS_FNAME   cblas_gemm_s8u8s32
#define  FNAME         gemm_s8u8s32
#elif  defined S8U8S32_EXPLICIT
#define  CBLAS_FNAME   cblas_gemm_s8u8s32_explicit
#define  FNAME         gemm_s8u8s32_explicit
#elif defined S8S8S32
#define  CBLAS_FNAME   cblas_gemm_s8s8s32
#define  FNAME         gemm_s8s8s32
#else
#error "undefined FNAME or CBLAS_FNAME"
#endif
#endif

#if defined S8U8S32 || defined S8U8S32_EXPLICIT
#define  A_TYPE        MKL_INT8
#define  B_TYPE        MKL_UINT8
#elif defined S8S8S32
#define  A_TYPE        MKL_INT8
#define  B_TYPE        MKL_INT8
#endif

void CBLAS_FNAME
(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
 const CBLAS_OFFSET offsetc, const MKL_INT m, const MKL_INT n, const MKL_INT k,
 const float alpha, const void *a, const MKL_INT lda, const MKL_INT8 oa,
 const void *b, const MKL_INT ldb, const MKL_INT8 ob, const float beta,
 MKL_INT32 *c, const MKL_INT ldc, const MKL_INT32 *oc)
{
#ifdef LIB_BUILD
#if defined S8U8S32
  printf ("Calling gemm_s8u8s32_explicit from cblas_gemm_s8u8s32\n"); fflush(0);
#elif defined S8S8S32
  printf ("Calling gemm_s8s8s32 from cblas_gemm_s8s8s32\n"); fflush(0);
#endif
#endif
  char transpose_a, transpose_b, off_type;

  if (transa == CblasNoTrans) {
    transpose_a = 'n';
  } else if (transa == CblasTrans) {
    transpose_a = 't';
  } else {
    printf ("ERROR: Unknown Transa parameter!\n"); fflush(0);
  }

  if (transb == CblasNoTrans) {
    transpose_b = 'n';
  } else if (transb == CblasTrans) {
    transpose_b = 't';
  } else {
    printf ("ERROR: Unknown Transb parameter!\n"); fflush(0);
  }

  if (Layout == CblasRowMajor) {
    if (offsetc == CblasRowOffset) {
      off_type = 'c';
    } else if (offsetc == CblasColOffset) {
      off_type = 'r';
    } else if (offsetc == CblasFixOffset) {
      off_type = 'f';
    } else {
      printf ("ERROR: Unknown offsetc parameter!\n"); fflush(0);
    }
    /* FNAME(&transpose_b, &transpose_a, &off_type, &n, &m, &k, &alpha, (MKL_INT8*)b, &ldb, &ob, (MKL_UINT8*)a, &lda, &oa, &beta, c, &ldc, oc); */
    FNAME(&transpose_b, &transpose_a, &off_type, &n, &m, &k, &alpha, (A_TYPE*)b, &ldb, &ob, (B_TYPE*)a, &lda, &oa, &beta, c, &ldc, oc);
  } else if (Layout == CblasColMajor) {
    if (offsetc == CblasRowOffset) {
      off_type = 'r';
    } else if (offsetc == CblasColOffset) {
      off_type = 'c';
    } else if (offsetc == CblasFixOffset) {
      off_type = 'f';
    } else {
      printf ("ERROR: Unknown offsetc parameter!\n"); fflush(0);
    }
    FNAME(&transpose_a, &transpose_b, &off_type, &m, &n, &k, &alpha, (A_TYPE*)a, &lda, &oa, (B_TYPE*)b, &ldb, &ob, &beta, c, &ldc, oc);
  } else {
    printf ("ERROR: Unknown Layout parameter!\n"); fflush(0);
  }
}
