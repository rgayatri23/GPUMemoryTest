SHELL = /bin/sh

#==========================
# Files
#==========================
EXE = axpy.ex
SRC = *.cu

COMP = nvcc

CUDA_PATH?=$(CUDA_ROOT)

accel=no
UVM=no
debug=no

#==========================
# Compilers
#==========================
ifeq ($(BIG_PROBLEM),y)
	DEFINE += -DBIG_PROBLEM_SIZE
endif

ifeq ($(CHECK_CORRECTNESS),y)
	DEFINE += -DVERIFY_GPU_CORRECTNESS
endif

ifeq ($(TIMEMORY_PROFILE),y)
	DEFINE += -DUSE_TIMEMORY
endif

ifeq ($(COMP),intel)
CXX = icc
else ifeq ($(COMP),pgi)
	CXX=pgc++
else ifeq ($(COMP),clang)
	CXX=clang++
else ifeq ($(COMP),nvcc)
	CXX=nvcc
endif

ifeq ($(PINNED_MEMORY),y)
	DEFINE += -DUSE_PINNED_MEMORY
else ifeq ($(MANAGED_MEMORY),y)
	DEFINE += -DUSE_MANAGED_MEMORY
else ifeq ($(ZERO_COPY),y)
	DEFINE += -DUSE_ZERO_COPY
else
	DEFINE += -DUSE_HOST_PAGEABLE_AND_DEVICE_MEMORY
endif

#==========================
# Machine specific info
# compilers and options
#==========================
ifeq ($(CXX),clang++)

	CXXFLAGS = -O2 -ffast-math -ffp-contract=fast -fstrict-aliasing -Wall -Wno-unused-variable
	CXXFLAGS += $(DEFINE)
	CXXFLAGS += -std=c++11
	CXXFLAGS += -lm
	ifeq ($(OPENMP),y)
		CXXFLAGS += -fopenmp
	endif
	ifeq ($(OPENMP_TARGET),y)
		CXXFLAGS += -fopenmp
		CXXFLAGS += -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_PATH} -ffp-contract=fast
		CXXFLAGS += -D__NO_MATH_INLINES -U__SSE2_MATH__ -U__SSE_MATH__
	endif
endif

ifeq ($(CXX),icc)
	CXXFLAGS = -O0 -g $(DEFINE) -std=c++11 -w
	CXXFLAGS += -qopt-report=5 #Opt Report
	ifeq ($(CRAY_CPU_TARGET),haswell)
		CXXFLAGS += -xcore-avx2
	else ifeq ($(CRAY_CPU_TARGET),mic-knl)
		CXXFLAGS += -xmic-avx512
	endif
endif

ifeq ($(CXX),pgc++)
	CXXFLAGS = -O3 -fast -Mlarge_arrays $(DEFINE) -std=c++11
	ifeq ($(OPENACC),y)
		CXXFLAGS += -acc
		CXXFLAGS += -ta=tesla:cc70
		CXXFLAGS += -Mcuda
		CXXFLAGS += -lnvToolsExt
		LDFLAGS = -acc -Mcuda -ta=tesla:cc70
		ifeq ($(accel),y)
			CXXFLAGS += -Minfo=accel
		endif
		ifeq ($(UVM),y)
			CXXFLAGS += -ta=tesla:managed
		else
			CXXFLAGS += -ta=tesla:pinned
		endif
	endif
endif

ifeq ($(CXX),nvcc)
#	CXXFLAGS = -I$(CUDA_PATH)/include
	CXXFLAGS = -I$(CUDA_PATH)/samples/common/inc/
	ifeq ($(TIMEMORY_PROFILE),y)
		CXXFLAGS += -I /project/projectdirs/m1759/timemory/corigpu/include
		CXXFLAGS += -DTIMEMORY_USE_CUDA
	endif
	CXXFLAGS += $(DEFINE)
	CXXFLAGS += -O3 -std=c++11 -Wno-deprecated-gpu-targets
	CXXFLAGS += -arch=sm_70
endif

#LDFLAGS = -L/global/project/projectdirs/m1759/timemory/corigpu/lib64 -ltimemory -lctimemory
LDFLAGS =

#==========================
# Compiler commands
#==========================
CXXLD         = $(CXX) $(CXXFLAGS) $(LDFLAGS)


#==========================
# Make the executable
#==========================
$(EXE): $(SRC) $(INC)
	echo $(SRC)
	$(CXXLD) $(SRC) -o $(EXE)
#==========================
#remove all objs
#==========================
clean:
	/bin/rm -f *.o $(EXE)
