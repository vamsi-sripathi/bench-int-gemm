#!/bin/bash

BINARY=igemm_s8s8s32.bin
export KMP_AFFINITY=granularity=fine,compact,1,0
export MKL_ENABLE_INSTRUCTIONS=AVX512_E1
NUMA_DOMAIN="-l"

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
          numactl ${NUMA_DOMAIN} ./${BINARY} $ta $tb f 128 258 198 $oa $ob
        done
      done
    done
  done
}

function cblas_gemm_transformer_set()
{
numactl ${NUMA_DOMAIN} ./${BINARY} n n f 10240 512 512 0 0
numactl ${NUMA_DOMAIN} ./${BINARY} n n f 1141 2048 512 0 0
numactl ${NUMA_DOMAIN} ./${BINARY} n n f 128 2048 512 0 0
numactl ${NUMA_DOMAIN} ./${BINARY} n n f 128 512 512 0 0
numactl ${NUMA_DOMAIN} ./${BINARY} n n f 2560 512 512 0 0
numactl ${NUMA_DOMAIN} ./${BINARY} n t f 128 33708 512 0 0
}

function blas_gemm_transformer_set()
{
numactl ${NUMA_DOMAIN} ./${BINARY} n n f 512 10240 512 0 0
numactl ${NUMA_DOMAIN} ./${BINARY} n n f 2048 1141 512 0 0
numactl ${NUMA_DOMAIN} ./${BINARY} n n f 2048 128 512 0 0
numactl ${NUMA_DOMAIN} ./${BINARY} n n f 512 128 512 0 0
numactl ${NUMA_DOMAIN} ./${BINARY} n n f 512 2560 512 0 0
numactl ${NUMA_DOMAIN} ./${BINARY} t n f 33708 128 512 0 0
}

# blas_gemm_transformer_set
# gemm_validate_sweep
# BINARY=cblas_igemm_s8s8s32.bin
# cblas_gemm_transformer_set
BINARY=cblas_igemm_s8u8s32_exp.bin
cblas_gemm_transformer_set
