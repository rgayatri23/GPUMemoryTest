SHELL = /bin/sh

#==========================
# Files
#==========================
EXE = axpy.ex
SRC = *.cu

CXX = nvcc


ifeq ($(LMOD_SYSTEM_NAME),summit)
CUDA_PATH?=$(OLCF_CUDA_ROOT)
DEFINE += -DON_SUMMIT
else
CUDA_PATH?=$(CUDA_ROOT)
endif

CHECK_CORRECTNESS = n
TIMEMORY_PROFILE = n
STRIDED = n
tUVM_prefetch = n

#==========================
# Compilers
#==========================

ifeq ($(CHECK_CORRECTNESS),y)
	DEFINE += -DVERIFY_GPU_CORRECTNESS
endif

ifeq ($(STRIDED),y)
	DEFINE += -DSTRIDED
endif

ifeq ($(tUVM_prefetch),y)
	DEFINE += -DtUVM_prefetch
endif

ifeq ($(TIMEMORY_PROFILE),y)
	DEFINE += -DUSE_TIMEMORY
endif

ifeq ($(PINNED_MEMORY),y)
	DEFINE += -DUSE_PINNED_MEMORY
else ifeq ($(MANAGED_MEMORY),y)
	DEFINE += -DUSE_MANAGED_MEMORY
else ifeq ($(ZERO_COPY),y)
	DEFINE += -DUSE_ZERO_COPY
else ifeq ($(PAGEABLE_MEMORY),y)
	DEFINE += -DUSE_HOST_PAGEABLE_AND_DEVICE_MEMORY
else
	DEFINE += -DRUN_ALL
endif

ifeq ($(CXX),nvcc)
#	CXXFLAGS = -I$(CUDA_PATH)/include
	CXXFLAGS = -I$(CUDA_PATH)/samples/common/inc/
	ifeq ($(TIMEMORY_PROFILE),y)
		CXXFLAGS += -I/project/projectdirs/m1759/timemory/corigpu/include
		CXXFLAGS += -DTIMEMORY_USE_CUDA
	endif
	CXXFLAGS += $(DEFINE)
	CXXFLAGS += -O3 -std=c++11 -Wno-deprecated-gpu-targets
	CXXFLAGS += -arch=sm_70
	CXXFLAGS += -lnvToolsExt
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
