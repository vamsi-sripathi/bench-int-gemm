#!/bin/bash

# BINARY=igemm_s8u8s32.bin
BINARY=cblas_igemm_s8u8s32_exp.bin
export MKL_ENABLE_INSTRUCTIONS=AVX512_E1
export LD_LIBRARY_PATH=$HOME/mkl_2019.0/mklml_lnx_2019.0.20180710/lib/:$LD_LIBRARY_PATH 

# BINARY=igemm_s8u8s32.bin
# export MKL_ENABLE_INSTRUCTIONS=AVX512
# export LD_LIBRARY_PATH=$HOME/mkl_lnx32e_20181025/__release_lnx/compiler/lib/intel64/:$LD_LIBRARY_PATH

export KMP_AFFINITY=granularity=fine,compact,1,0
NUMA_DOMAIN="-m0"

function gemm_sweep()
{
for n in {128..8192..64};
do
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f $n $n $n 0 0
done
}

function gemm_target_sizes()
{

# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 10240 n = 512 k = 512 oa = -128 ob = 0
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 10240 n = 512 k = 512 oa = -128 ob = 1
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 1054 n = 2048 k = 512 oa = -127 ob = 0
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 1054 n = 2048 k = 512 oa = -128 ob = 0
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 1054 n = 2048 k = 512 oa = -128 ob = 1
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 1141 n = 2048 k = 512 oa = -127 ob = 0
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 1141 n = 2048 k = 512 oa = -128 ob = 0
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 1141 n = 2048 k = 512 oa = -128 ob = 1
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 128 n = 2048 k = 512 oa = -127 ob = 0
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 128 n = 2048 k = 512 oa = -128 ob = 0
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 128 n = 2048 k = 512 oa = -128 ob = 1
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 128 n = 512 k = 512 oa = -127 ob = 0
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 128 n = 512 k = 512 oa = -127 ob = 1
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 128 n = 512 k = 512 oa = -128 ob = 0
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 128 n = 512 k = 512 oa = -128 ob = 1
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 2144 n = 512 k = 512 oa = -127 ob = 0
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 2144 n = 512 k = 512 oa = -128 ob = 0
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 2144 n = 512 k = 512 oa = -128 ob = 1
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 2560 n = 512 k = 512 oa = -127 ob = 0
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 2560 n = 512 k = 512 oa = -128 ob = 0
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 2560 n = 512 k = 512 oa = -128 ob = 1
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 8576 n = 512 k = 512 oa = -128 ob = 0
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 0 m = 8576 n = 512 k = 512 oa = -128 ob = 1
# INTEL_DEBUG: Executing MKL Optimized IGEMM: ta = 0 tb = 1 m = 128 n = 33708 k = 512 oa = -128 ob = 0
# 
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 512 10240 512 0 -128
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 512 10240 512 1 -128

  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 2048 1054 512 0 -127
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 2048 1054 512 0 -128
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 2048 1054 512 1 -128

  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 2048 1141 512  0 -127
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 2048 1141 512  0 -128
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 2048 1141 512  1 -128

  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 2048 128 512  0 -127
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 2048 128 512  0 -128
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 2048 128 512  1 -128

  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 512 128 512  0 -127
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 512 128 512  0 -128
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 512 128 512  1 -128

  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 512 2144 512  0 -127
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 512 2144 512  0 -128
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 512 2144 512  1 -128

  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 512 2560 512  0 -127
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 512 2560 512  0 -128
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 512 2560 512  1 -128

  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 512 8576 512 0 -128
  numactl ${NUMA_DOMAIN} ./${BINARY} n n f 512 8576 512 1 -128

  numactl ${NUMA_DOMAIN} ./${BINARY} t n f 33708 128 512 0 -128


  # numactl ${NUMA_DOMAIN} ./${BINARY} n n f 512 128 512  -128 1
}

# gemm_target_sizes

function cblas_gemm_target_sizes()
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

cblas_gemm_target_sizes
