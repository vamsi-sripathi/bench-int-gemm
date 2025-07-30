CC    = icc
COPTS = -std=c99 -Wall -O3 -qopenmp -DVALIDATION -UPRINT_MATRIX -UFILL_MATRIX -qopt-zmm-usage=high

ifdef PAD_LD
COPTS += -DPAD_LD
endif

# MKLROOT=$(HOME)/mkl_2019.1.144
# MKLROOT=$(HOME)/mkl_lnx32e_20181025/__release_lnx/mkl/

ifndef MKLROOT
$(error "MKLROOT is undefined")
endif

ifdef USE_MKLML
# MKLROOT     = /ec/site/disks/aipg_lab_home_pool_01/vsripath/.cache/bazel/_bazel_vsripath/18f29dcd942cbec2bdfd03b11af9b938/external/mkl_linux/
MKLROOT     = $(HOME)/mkl_2019.0/mklml_lnx_2019.0.20180710/
MKL_LIB_DIR = $(MKLROOT)/lib
MKL_LIBS    = -L$(MKL_LIB_DIR)/ -lmklml_intel -liomp5
else
MKL_LIB_DIR = $(MKLROOT)/lib/intel64
# MKL_LIB_DIR = $(MKLROOT)/lib/intel64_lin
MKL_LIBS    = -Wl,--start-group $(MKL_LIB_DIR)/libmkl_intel_lp64.a $(MKL_LIB_DIR)/libmkl_intel_thread.a $(MKL_LIB_DIR)/libmkl_core.a -Wl,--end-group
# MKL_LIBS    = -L$(MKL_LIB_DIR)/ -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
endif

MKL_INC     = $(MKLROOT)/include
#AUX_LIBS    = -lpthread -lm

ALL_LIBS    = $(MKL_LIBS) $(AUX_LIBS)

SRC = igemm_bench.c
OBJ = igemm_s8u8s32_bench.o igemm_s8u8s32_exp_bench.o igemm_s8u8s32_exp.o igemm_s8u8s32_ref.o igemm_s8u8s32_patched_bench.o igemm_s8u8s32_patched.o

IGEMM_S8U8S32_DEV_OBJS  = igemm_s8u8s32_dev_bench.o igemm_s8u8s32_ref.o
IGEMM_S8U8S32_DEV_OBJS += gemm_s8u8s32_dev_driver.o gemm_s8u8s32_dev_an_copy.o gemm_s8u8s32_dev_bn_copy.o gemm_s8u8s32_dev_ker_16x4.o gemm_s8u8s32_dev_ker_16x8.o
igemm_s8u8s32_dev.bin: $(IGEMM_S8U8S32_DEV_OBJS)
	$(CC) $(COPTS) -o $@ $^ $(ALL_LIBS)
igemm_s8u8s32_dev_bench.o: igemm_bench.c
	$(CC) -c -DS8U8S32 $(COPTS) -I$(MKL_INC) -o $@ $<
gemm_s8u8s32_dev_driver.o: ./avx512_gemm/gemm_s8u8s32_dev_driver.c
	$(CC) -c -xCORE-AVX512 -DS8U8S32 $(COPTS) -I$(MKL_INC) -o $@ $<
gemm_s8u8s32_dev_an_copy.o: ./avx512_gemm/gemm_s8u8s32_dev_an_copy.c
	$(CC) -c -xCORE-AVX512 -DS8U8S32 $(COPTS) -I$(MKL_INC) -o $@ $<
gemm_s8u8s32_dev_bn_copy.o: ./avx512_gemm/gemm_s8u8s32_dev_bn_copy.c
	$(CC) -c -xCORE-AVX512 -DS8U8S32 $(COPTS) -I$(MKL_INC) -o $@ $<
gemm_s8u8s32_dev_ker_16x4.o: ./avx512_gemm/gemm_s8u8s32_dev_ker_16x4.c
	$(CC) -c -xCORE-AVX512 -DS8U8S32 $(COPTS) -I$(MKL_INC) -o $@ $< #-S -fcode-asm 
gemm_s8u8s32_dev_ker_16x8.o: ./avx512_gemm/gemm_s8u8s32_dev_ker_16x8.c
	$(CC) -c -xCORE-AVX512 -DS8U8S32 $(COPTS) -I$(MKL_INC) -o $@ $< #-S -fcode-asm 


igemm_s8u8s32.bin: igemm_s8u8s32_bench.o igemm_s8u8s32_ref.o
	$(CC) $(COPTS) -o $@ $^ $(ALL_LIBS)
igemm_s8u8s32_bench.o: igemm_bench.c
	$(CC) -c -DS8U8S32 $(COPTS) -I$(MKL_INC) -o $@ $<
igemm_s8u8s32_ref.o: igemm_ref.c
	$(CC) -c -xCORE-AVX512 -DS8U8S32 $(COPTS) -I$(MKL_INC) -o $@ $<


igemm_s8u8s32_exp.bin: igemm_s8u8s32_exp_bench.o igemm_s8u8s32_exp.o igemm_s8u8s32_ref.o
	$(CC) $(COPTS) -o $@ $^ $(ALL_LIBS)
igemm_s8u8s32_exp_bench.o: igemm_bench.c
	$(CC) -c -DS8U8S32_EXPLICIT $(COPTS) -I$(MKL_INC) -o $@ $<
igemm_s8u8s32_exp.o: igemm_s8u8s32_explicit.c
	$(CC) -c -xCORE-AVX512 $(COPTS) -I$(MKL_INC) -o $@ $<

igemm_s8s8s32.bin: igemm_s8s8s32_bench.o igemm_s8s8s32.o igemm_s8s8s32_ref.o #igemm_s8u8s32_exp.o 
	$(CC) $(COPTS) -o $@ $^ $(ALL_LIBS)
igemm_s8s8s32_bench.o: igemm_bench.c
	$(CC) -c -DS8S8S32 $(COPTS) -I$(MKL_INC) -o $@ $<
igemm_s8s8s32.o: igemm_s8s8s32.c
	$(CC) -c -xCORE-AVX512 -DS8S8S32 $(COPTS) -I$(MKL_INC) -o $@ $<
igemm_s8s8s32_ref.o: igemm_ref.c
	$(CC) -c -xCORE-AVX512 -DS8S8S32 $(COPTS) -I$(MKL_INC) -o $@ $<

cblas_igemm_s8u8s32_exp.bin: cblas_igemm_s8u8s32_exp_bench.o cblas_igemm_s8u8s32_exp.o igemm_s8u8s32_ref.o igemm_s8u8s32_exp.o
	$(CC) $(COPTS) -o $@ $^ $(ALL_LIBS)
cblas_igemm_s8u8s32_exp_bench.o: igemm_bench.c
	$(CC) -c -DBENCH_CBLAS -DS8U8S32_EXPLICIT $(COPTS) -I$(MKL_INC) -o $@ $<
cblas_igemm_s8u8s32_exp.o: cblas_igemm.c
	$(CC) -c -xCORE-AVX512 -DS8U8S32_EXPLICIT $(COPTS) -I$(MKL_INC) -o $@ $<

cblas_igemm_s8s8s32.bin: cblas_igemm_s8s8s32_bench.o cblas_igemm_s8s8s32.o igemm_s8s8s32_ref.o igemm_s8s8s32.o igemm_s8u8s32_exp.o
	$(CC) $(COPTS) -o $@ $^ $(ALL_LIBS)
cblas_igemm_s8s8s32_bench.o: igemm_bench.c
	$(CC) -c -DBENCH_CBLAS -DS8S8S32 $(COPTS) -I$(MKL_INC) -o $@ $<
cblas_igemm_s8s8s32.o: cblas_igemm.c
	$(CC) -c -xCORE-AVX512 -DS8S8S32 $(COPTS) -I$(MKL_INC) -o $@ $<

tf_cblas_igemm_s8u8s32.bin: cblas_igemm_s8u8s32_bench.o igemm_s8u8s32_ref.o libcblas_gemm_s8u8s32_exp.so
	$(CC) $(COPTS) -o $@ $^ $(ALL_LIBS)
cblas_igemm_s8u8s32_bench.o: igemm_bench.c
	$(CC) -c -DBENCH_CBLAS -DS8U8S32 $(COPTS) -I$(MKL_INC) -o $@ $<

tf_cblas_igemm_s8s8s32.bin: cblas_igemm_s8s8s32_bench.o igemm_s8s8s32_ref.o libcblas_gemm_s8s8s32.so
	$(CC) $(COPTS) -o $@ $^ $(ALL_LIBS)

libcblas_gemm_s8u8s32_exp.so: cblas_igemm_s8u8s32_libbuild.o igemm_s8u8s32_exp_fpic.o
	$(CC) -shared -fPIC $(COPTS) -o $@ $^ $(ALL_LIBS)
igemm_s8u8s32_exp_fpic.o: igemm_s8u8s32_explicit.c
	$(CC) -fPIC -c -xCORE-AVX512 $(COPTS) -I$(MKL_INC) -o $@ $<
cblas_igemm_s8u8s32_libbuild.o: cblas_igemm.c
	$(CC) -fPIC -c -xCORE-AVX512 -DLIB_BUILD -DS8U8S32 $(COPTS) -I$(MKL_INC) -o $@ $<

libcblas_gemm_s8s8s32.so: cblas_igemm_s8s8s32_libbuild.o igemm_s8s8s32_fpic.o igemm_s8u8s32_exp_fpic.o
	$(CC) -shared -fPIC $(COPTS) -o $@ $^ $(ALL_LIBS)
igemm_s8s8s32_fpic.o: igemm_s8s8s32.c
	$(CC) -fPIC -c -xCORE-AVX512 $(COPTS) -I$(MKL_INC) -o $@ $<
cblas_igemm_s8s8s32_libbuild.o: cblas_igemm.c
	$(CC) -fPIC -c -xCORE-AVX512 -DLIB_BUILD -DS8S8S32 $(COPTS) -I$(MKL_INC) -o $@ $<

all: igemm_s8u8s32.bin igemm_s8u8s32_exp.bin cblas_igemm_s8u8s32_exp.bin igemm_s8s8s32.bin cblas_igemm_s8s8s32.bin tf_cblas_igemm_s8u8s32.bin tf_cblas_igemm_s8s8s32.bin
lib: libcblas_gemm_s8u8s32_exp.so libcblas_gemm_s8s8s32.so
run: igemm_s8u8s32.bin
	KMP_AFFINITY=granularity=fine,compact,1,0 numactl -i all ./igemm_s8u8s32.bin n n f 512 512 512 0 0

clean:
	rm -rf $(OBJ) *.bin *.o

.PHONY: all clean run
