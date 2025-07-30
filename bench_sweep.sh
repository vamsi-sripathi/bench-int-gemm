#!/bin/bash

export KMP_AFFINITY=granularity=fine,compact,1,0
export MKL_ENABLE_INSTRUCTIONS=AVX512_E1

function gemm_validate_sweep()
{
  for ta in n t;
  do
    for tb in n t;
    do
      for oa in 0 10;
      do
        for ob in 0 10;
        do
          ./${BINARY} $ta $tb f 128 258 198 $oa $ob
        done
      done
    done
  done
}

for bin in igemm_s8u8s32.bin igemm_s8u8s32_exp.bin cblas_igemm_s8u8s32_exp.bin igemm_s8s8s32.bin cblas_igemm_s8s8s32.bin tf_cblas_igemm_s8s8s32.bin  tf_cblas_igemm_s8u8s32.bin;
do
   BINARY=${bin}
   gemm_validate_sweep
done
