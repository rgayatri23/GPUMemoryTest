
/*
 * AXPY :
 * Y += A*X
 */

//#include <iostream>
#include <strings.h>
#include <stdio.h>
#include <stdlib.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include <sys/time.h>
#if USE_TIMEMORY
#include <timemory/timemory.hpp>
#endif
#include <sys/time.h>

using namespace std;

#define check_num_values 10

//Handle 2-dim pointing
#define Y(i,j) Y[i*M + j]
#define X(i,j) X[i*M + j]

#define dataType double
#define CUDA_HOSTDEV __host__ __device__

inline double elapsedTime(timeval start_time, timeval end_time)
{
  return ((end_time.tv_sec - start_time.tv_sec) +1e-6*(end_time.tv_usec - start_time.tv_usec));
}


void PreComputeFewValues(int N, int M, dataType a, dataType *Y, dataType *X)
{
  if(N >= check_num_values)
    for(int i = 0; i < check_num_values; ++i)
        for(int j = 0; j < M; ++j)
            Y(i,j) += a * X(i,j);
}

void checkGPUCorrectness(int N, int M, dataType *Y, dataType *yOrig)
{
    dataType error = 0.0;
    if(N >= check_num_values)
        for(int i = 0; i < check_num_values; ++i)
        {
            dataType diff = yOrig[i] - Y[i];
            error += diff*diff;
        }

    if(error < 0.0001)
        cout << "Successfull completion of GPU kernels with expected output\n" << endl;
    else
        cout << "Unsuccessfull completion of GPU kernels with diverging results\n" << endl;
}

__global__ void axpyKernel(int N, int M, dataType a, dataType *Y, dataType *X)
{
    for(int i = blockIdx.x; i < N; i+=gridDim.x)
        for(int j = threadIdx.x; j < M; j+=blockDim.x)
            Y(i,j) += a * X(i,j);
}

int main(int argc, char **argv)
{
#if USE_TIMEMORY
  tim::timemory_init(argc, argv);
  tim::settings::timing_precision() = 6;
  tim::settings::timing_scientific() = true;
#endif

  int N = 10, M = 10;
  if(argc > 1)
  {
      if(argc == 3)
      {
          N = atoi(argv[1]);
          M = atoi(argv[2]);
      }
      else
      {
          cout << "Input format is ./axpy.ex N M \n" << endl;
          exit(EXIT_FAILURE);
      }
  }

  double elapsed_total, elapsed_memAlloc, elapsed_kernel;
  timeval startTotalTimer, endTotalTimer,
          startMemAllocTimer, endMemAllocTimer,
          startKernelTimer, endKernelTimer;
  gettimeofday(&startTotalTimer, NULL);

  cout << "M = " << M << "\t N = " << N << endl;
  cout << "Total Memory Footprint = " << (size_t)(M*N*sizeof(dataType)/(1024*1024*1024)) << " GBs" << endl;
  cout << "threadblocks = " << N << "  and data accessed by each threadblock = " << M*sizeof(double) << " bytes" << endl;

  int device;
//    checkCudaErrors(cudaSetDevice(1));
  checkCudaErrors(cudaGetDevice(&device));
  cout << "Device number = " << device << endl;

  dataType *Y, *X;
  dataType *d_Y, *d_X;
  bool copyFlag = false;

  dim3 grid(N,1,1);
  dim3 threads(M,1,1);

  dataType a = 0.5;
  dataType rand1 = (dataType)rand() / (dataType)RAND_MAX;

#if USE_TIMEMORY
  using namespace tim::component;
  using auto_tuple_t = tim::auto_tuple<cuda_event, real_clock>;
//    using auto_tuple_t = tim::auto_tuple<real_clock, cpu_clock, cpu_util, peak_rss, cuda_event, system_clock, user_clock, cpu_clock>;
  using comp_tuple_t = typename auto_tuple_t::component_type;
  comp_tuple_t measure("total", true);
  comp_tuple_t data_allocation("data_movement", true);
  measure.start();
  data_allocation.start();
#endif

  //Allocating data
  gettimeofday(&startMemAllocTimer, NULL);
#if defined(USE_HOST_PAGEABLE_AND_DEVICE_MEMORY)
  printf("###############Using HOST_PAGEABLE_AND_DEVICE_MEMORY###############\n");
  X = (dataType*) malloc(N*M*sizeof(dataType));
  Y = (dataType*) malloc(N*M*sizeof(dataType));

  //Allocate memory on device
  checkCudaErrors(cudaMalloc(&d_X, N*M*sizeof(dataType)));
  checkCudaErrors(cudaMalloc(&d_Y, N*M*sizeof(dataType)));

  copyFlag = true; //Switch on the copy flag

#elif defined(USE_PINNED_MEMORY)
  printf("###############Using PINNED_MEMORY###############\n");
  checkCudaErrors(cudaMallocHost(&X, N*M*sizeof(dataType)));
  checkCudaErrors(cudaMallocHost(&Y, N*M*sizeof(dataType)));

    //Allocate memory on device
  checkCudaErrors(cudaMalloc(&d_X, N*M*sizeof(dataType)));
  checkCudaErrors(cudaMalloc(&d_Y, N*M*sizeof(dataType)));

  copyFlag = true; //Switch on the copy flag

#elif defined(USE_MANAGED_MEMORY)
  printf("###############Using MANAGED_MEMORY###############\n");
  checkCudaErrors(cudaMallocManaged(&d_X, N*M*sizeof(dataType)));
  checkCudaErrors(cudaMallocManaged(&d_Y, N*M*sizeof(dataType)));
  X = d_X; Y = d_Y;

#elif defined(USE_ZERO_COPY)
  printf("###############Using ZERO_COPY###############\n");
  checkCudaErrors(cudaMallocHost(&X, N*M*sizeof(dataType)));
  checkCudaErrors(cudaMallocHost(&Y, N*M*sizeof(dataType)));
  checkCudaErrors(cudaHostGetDevicePointer(&d_X,X,0));
  checkCudaErrors(cudaHostGetDevicePointer(&d_Y,Y,0));
#endif
  gettimeofday(&endMemAllocTimer, NULL);

  memset(Y,0,N*M*sizeof(dataType));
  for(int i = 0; i < N; ++i)
      for(int j = 0; j < M; ++j)
          X[i*M + j] = rand1 * (i+1);

  if(copyFlag == true)
      checkCudaErrors(cudaMemcpy(d_X, X, N*M*sizeof(dataType), cudaMemcpyHostToDevice));

#if VERIFY_GPU_CORRECTNESS
  dataType *yOrig;
  yOrig = (dataType*) malloc(check_num_values*M*sizeof(dataType));
  PreComputeFewValues(N,M,a,yOrig,X);
#endif

  //Actual CUDA kernel
  gettimeofday(&startKernelTimer, NULL);
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);

  checkCudaErrors(cudaDeviceSynchronize());
  if(copyFlag)
      checkCudaErrors(cudaMemcpy(Y, d_Y, N*M*sizeof(dataType), cudaMemcpyDeviceToHost));
  gettimeofday(&endKernelTimer, NULL);

#if USE_TIMEMORY
  data_allocation.stop();
#endif


#if VERIFY_GPU_CORRECTNESS
  checkGPUCorrectness(N,M,Y,yOrig);
#endif

#if defined (USE_HOST_PAGEABLE_AND_DEVICE_MEMORY)
    free(X);
    free(Y);
    checkCudaErrors(cudaFree(d_X));
    checkCudaErrors(cudaFree(d_Y));
#elif defined(USE_PINNED_MEMORY)
    checkCudaErrors(cudaFreeHost(X));
    checkCudaErrors(cudaFreeHost(Y));
    checkCudaErrors(cudaFree(d_X));
    checkCudaErrors(cudaFree(d_Y));
#elif defined(USE_MANAGED_MEMORY)
    checkCudaErrors(cudaFree(d_X));
    checkCudaErrors(cudaFree(d_Y));
#elif defined(USE_ZERO_COPY)
    checkCudaErrors(cudaFreeHost(X));
    checkCudaErrors(cudaFreeHost(Y));
#endif

#if USE_TIMEMORY
    measure.stop();
#endif
  gettimeofday(&endTotalTimer, NULL);

  //calculate elapsed time
  elapsed_total = elapsedTime(startTotalTimer, endTotalTimer);
  elapsed_memAlloc = elapsedTime(startMemAllocTimer, endMemAllocTimer);
  elapsed_kernel = elapsedTime(startKernelTimer, endKernelTimer);

  cout << "************ MemAlloc-time = " << elapsed_memAlloc << " [sec] ************\n" << endl;
  cout << "************ Kernel-time = " << elapsed_kernel << " [sec] ************\n" << endl;
  cout << "************ Total-time = " << elapsed_total << " [sec] ************\n" << endl;
    return 0;
}
