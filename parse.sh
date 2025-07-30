#!/bin/bash

# gemm_s8u8s32_explicit: transa= n, transb= n, off_type= f, m= 512, n= 10240, k= 512, lda= 512, ldb= 512, ldc= 512, off_a= 1, off_b= -128, avg-gflops= 10180.50, best-gflops= 10461.73, 
awk '/gflops/{print $3"-"$5"-"$7"-"$9"-"$11"-"$13"-"$21"-"$23" "$25}' $1 | tr -d ","
