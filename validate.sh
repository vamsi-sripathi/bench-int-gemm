#!/bin/bash

# BINARY=igemm_s8u8s32.bin
# export MKL_ENABLE_INSTRUCTIONS=AVX512_E1

export KMP_AFFINITY=granularity=fine,compact,1,0
NUMA_DOMAIN="-m0"

function gemm_validate()
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

function gemm_target_sizes()
{
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 10240 512 512  -128 0
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 10240 512 512  -128 1

  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 1054 2048 512  -127 0
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 1054 2048 512  -128 0
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 1054 2048 512  -128 1

  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 1141 2048 512  -127  0
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 1141 2048 512  -128  0
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 1141 2048 512  -128  1

  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 128 2048 512  -127  0
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 128 2048 512  -128  0
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 128 2048 512  -128  1

  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 128 512 512  -127  0
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 128 512 512  -128  0
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 128 512 512  -128  1

  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 2144 512 512  -127  0
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 2144 512 512  -128  0
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 2144 512 512  -128  1

  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 512 2560 512  -127  0
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 512 2560 512  -128  0
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 512 2560 512  -128  1

  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 8576 512 512 -128 0
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 8576 512 512 -128 1

  numactl ${NUMA_DOMAIN} ./${BINARY} n t f 128 33708 512 -128 0
}

BINARY=cblas_igemm_s8u8s32_exp.bin
gemm_validate
gemm_target_sizes

BINARY=igemm_s8u8s32_exp.bin
gemm_validate

BINARY=igemm_s8u8s32.bin
gemm_validate
