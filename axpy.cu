
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
#include <nvToolsExt.h>
#if USE_TIMEMORY
#include <timemory/timemory.hpp>
#endif
#include <sys/time.h>

using namespace std;

#define check_num_values 10
#define NUM_LOOPS 5
#define run_kernel_times 5
bool print_csv = false;

//Handle 2-dim pointing
#define Y(i,j) Y[i*M + j]
#define X(i,j) X[i*M + j]

#define dataType double
#define CUDA_HOSTDEV __host__ __device__

inline double elapsedTime(timeval start_time, timeval end_time)
{
  return ((end_time.tv_sec - start_time.tv_sec) +1e-6*(end_time.tv_usec - start_time.tv_usec));
}


void PreComputeFewValues(int N, int M, dataType a, dataType rand1, dataType *Y, dataType *X)
{
  if(N >= check_num_values)
  {
    for(int i = 0; i < check_num_values; ++i)
      for(int j = 0; j < M; ++j)
          X(i,j) = rand1 * (i+1);

    for(int i = 0; i < check_num_values; ++i)
      for(int j = 0; j < M; ++j)
        Y(i,j) += a * X(i,j);
  }
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
  for(int iter = 0; iter < 10; ++iter)
    for(int i = blockIdx.x; i < N; i+=gridDim.x)
        for(int j = threadIdx.x; j < M; j+=blockDim.x)
            Y(i,j) += a * X(i,j);
}

void zero_copy(dataType a, dataType rand1, double& elapsed_memAlloc, double& elapsed_memcpy, double& elapsed_init, double& elapsed_init_kernel, double& elapsed_kernel, double& elapsed_total, int N, int M, dataType* yOrig)
{
  nvtxRangePushA("Zero_copy");
  int device;
  checkCudaErrors(cudaSetDevice(3));
  checkCudaErrors(cudaGetDevice(&device));
  timeval startMemAllocTimer, endMemAllocTimer,
          startInitTimer, endInitTimer,
          startKernelTimer, endKernelTimer, 
          startTotalTimer, endTotalTimer, 
          startInitKernel, endInitKernel;

  gettimeofday(&startTotalTimer, NULL);

  dataType *Y, *X;
  dataType *d_Y, *d_X;
  dim3 grid(N,1,1);
  dim3 threads(32,1,1);

  gettimeofday(&startMemAllocTimer, NULL);
  checkCudaErrors(cudaHostAlloc(&X, N*M*sizeof(dataType), cudaHostAllocDefault));
  checkCudaErrors(cudaHostAlloc(&Y, N*M*sizeof(dataType), cudaHostAllocDefault));
  checkCudaErrors(cudaHostGetDevicePointer(&d_X,X,0));
  checkCudaErrors(cudaHostGetDevicePointer(&d_Y,Y,0));
  gettimeofday(&endMemAllocTimer, NULL);

  gettimeofday(&startInitTimer, NULL);
  memset(Y,0,N*M*sizeof(dataType));
  for(int i = 0; i < N; ++i)
      for(int j = 0; j < M; ++j)
          X[i*M + j] = rand1 * (i+1);
  gettimeofday(&endInitTimer, NULL);

  //Run the kernel twice before any of the timings begin to get rid of the initial-hickups
#if !defined(VERIFY_GPU_CORRECTNESS)
  gettimeofday(&startInitKernel, NULL);
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  checkCudaErrors(cudaDeviceSynchronize());
  gettimeofday(&endInitKernel, NULL);
#endif

  //Start Kernel Timer
  gettimeofday(&startKernelTimer, NULL);

  //Start actual kernel//Start Kernel Timerpinned
  for(int iter = 0; iter < run_kernel_times; ++iter)
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  checkCudaErrors(cudaDeviceSynchronize());

  //End Kernel Timer
  gettimeofday(&endKernelTimer, NULL);

#if VERIFY_GPU_CORRECTNESS
  cout << "ZERO-COPY : \t" ;
  checkGPUCorrectness(N,M,Y,yOrig);
#endif

#if !defined(ON_SUMMIT)
  checkCudaErrors(cudaFreeHost(X));
  checkCudaErrors(cudaFreeHost(Y));
#endif

  gettimeofday(&endTotalTimer, NULL);
  elapsed_memAlloc += elapsedTime(startMemAllocTimer, endMemAllocTimer);
  elapsed_init += elapsedTime(startInitTimer, endInitTimer);
  elapsed_kernel += elapsedTime(startKernelTimer, endKernelTimer);
  elapsed_total += elapsedTime(startTotalTimer, endTotalTimer);
  elapsed_init_kernel += elapsedTime(startInitKernel, endInitKernel);
  nvtxRangePop();
}

#if defined(ON_SUMMIT)
void tUVM(dataType a, dataType rand1, double &elapsed_memAlloc, double& elapsed_memcpy, double& elapsed_init, double& elapsed_init_kernel, double &elapsed_kernel, double& elapsed_total, int N, int M, dataType *yOrig)
{
  nvtxRangePushA("tUVM");
  int device;
  checkCudaErrors(cudaSetDevice(4));
  checkCudaErrors(cudaGetDevice(&device));
  timeval startMemAllocTimer, endMemAllocTimer,
          startInitTimer, endInitTimer,
          startKernelTimer, endKernelTimer, 
          startTotalTimer, endTotalTimer, 
          startInitKernel, endInitKernel;

  gettimeofday(&startTotalTimer, NULL);
  dataType *d_Y, *d_X;
  dim3 grid(N,1,1);
  dim3 threads(32,1,1);

  gettimeofday(&startMemAllocTimer, NULL);
  
  d_X = (dataType*) malloc(N*M*sizeof(dataType));
  d_Y = (dataType*) malloc(N*M*sizeof(dataType));
  gettimeofday(&endMemAllocTimer, NULL);

  gettimeofday(&startInitTimer, NULL);
  for(int i = 0; i < N; ++i)
      for(int j = 0; j < M; ++j)
          d_X[i*M + j] = rand1 * (i+1);
  gettimeofday(&endInitTimer, NULL);

#if !defined(VERIFY_GPU_CORRECTNESS)
  //Run the kernel couple of times before the actual timings begins
  gettimeofday(&startInitKernel, NULL);
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  checkCudaErrors(cudaDeviceSynchronize());
  gettimeofday(&endInitKernel, NULL);
#endif

  //Start Kernel Timer
  gettimeofday(&startKernelTimer, NULL);

  //Start actual kernel//Start Kernel Timerpinned
  for(int iter = 0; iter < run_kernel_times; ++iter)
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  checkCudaErrors(cudaDeviceSynchronize());

  //End Kernel Timer
  gettimeofday(&endKernelTimer, NULL);

#if VERIFY_GPU_CORRECTNESS
  cout << "MANAGED-MEMORY : \t" ;
  checkGPUCorrectness(N,M,d_Y,yOrig);
#endif

  free(d_X);
  free(d_Y);

  gettimeofday(&endTotalTimer, NULL);

  elapsed_memAlloc += elapsedTime(startMemAllocTimer, endMemAllocTimer);
  elapsed_init += elapsedTime(startInitTimer, endInitTimer);
  elapsed_kernel += elapsedTime(startKernelTimer, endKernelTimer);
  elapsed_total += elapsedTime(startTotalTimer, endTotalTimer);
  elapsed_init_kernel += elapsedTime(startInitKernel, endInitKernel);
  nvtxRangePop();
}
#endif


void managed_memory(dataType a, dataType rand1, double &elapsed_memAlloc, double& elapsed_memcpy, double& elapsed_init, double& elapsed_init_kernel, double &elapsed_kernel, double& elapsed_total, int N, int M, dataType *yOrig)
{
  nvtxRangePushA("managed_memory");
  int device;
  checkCudaErrors(cudaSetDevice(2));
  checkCudaErrors(cudaGetDevice(&device));
  timeval startMemAllocTimer, endMemAllocTimer,
          startInitTimer, endInitTimer,
          startKernelTimer, endKernelTimer, 
          startTotalTimer, endTotalTimer, 
          startInitKernel, endInitKernel;

  gettimeofday(&startTotalTimer, NULL);
  dataType *d_Y, *d_X;
  dim3 grid(N,1,1);
  dim3 threads(32,1,1);

  gettimeofday(&startMemAllocTimer, NULL);
  
  checkCudaErrors(cudaMallocManaged(&d_X, N*M*sizeof(dataType)));
  checkCudaErrors(cudaMallocManaged(&d_Y, N*M*sizeof(dataType)));
  gettimeofday(&endMemAllocTimer, NULL);

  gettimeofday(&startInitTimer, NULL);
  for(int i = 0; i < N; ++i)
      for(int j = 0; j < M; ++j)
          d_X[i*M + j] = rand1 * (i+1);
  gettimeofday(&endInitTimer, NULL);

#if !defined(VERIFY_GPU_CORRECTNESS)
  //Run the kernel couple of times before the actual timings begins
  gettimeofday(&startInitKernel, NULL);
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  checkCudaErrors(cudaDeviceSynchronize());
  gettimeofday(&endInitKernel, NULL);
#endif

  //Start Kernel Timer
  gettimeofday(&startKernelTimer, NULL);

  //Start actual kernel//Start Kernel Timerpinned
  for(int iter = 0; iter < run_kernel_times; ++iter)
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  checkCudaErrors(cudaDeviceSynchronize());

  //End Kernel Timer
  gettimeofday(&endKernelTimer, NULL);

#if VERIFY_GPU_CORRECTNESS
  cout << "MANAGED-MEMORY : \t" ;
  checkGPUCorrectness(N,M,d_Y,yOrig);
#endif

  checkCudaErrors(cudaFree(d_X));
  checkCudaErrors(cudaFree(d_Y));

  gettimeofday(&endTotalTimer, NULL);

  elapsed_memAlloc += elapsedTime(startMemAllocTimer, endMemAllocTimer);
  elapsed_init += elapsedTime(startInitTimer, endInitTimer);
  elapsed_kernel += elapsedTime(startKernelTimer, endKernelTimer);
  elapsed_total += elapsedTime(startTotalTimer, endTotalTimer);
  elapsed_init_kernel += elapsedTime(startInitKernel, endInitKernel);
  nvtxRangePop();
}

void pinned_memory(dataType a, dataType rand1, double &elapsed_memAlloc, double& elapsed_memcpy, double& elapsed_init, double& elapsed_init_kernel, double &elapsed_kernel, double& elapsed_total, int N, int M, dataType* yOrig)
{
  nvtxRangePushA("pinned_memory");
  int device;
  checkCudaErrors(cudaSetDevice(1));
  checkCudaErrors(cudaGetDevice(&device));
  timeval startMemAllocTimer, endMemAllocTimer,
          startInitTimer, endInitTimer,
          startKernelTimer, endKernelTimer, 
          startTotalTimer, endTotalTimer, 
          startInitKernel, endInitKernel, 
          startMemCpyTimer, endMemCpyTimer;

  gettimeofday(&startTotalTimer, NULL);


  dataType *Y, *X;
  dataType *d_Y, *d_X;
  dim3 grid(N,1,1);
  dim3 threads(32,1,1);

  gettimeofday(&startMemAllocTimer, NULL);
  checkCudaErrors(cudaHostAlloc(&X, N*M*sizeof(dataType), cudaHostAllocMapped));
  checkCudaErrors(cudaHostAlloc(&Y, N*M*sizeof(dataType), cudaHostAllocMapped));

    //Allocate memory on device
  checkCudaErrors(cudaMalloc(&d_X, N*M*sizeof(dataType)));
  checkCudaErrors(cudaMalloc(&d_Y, N*M*sizeof(dataType)));
  gettimeofday(&endMemAllocTimer, NULL);

  gettimeofday(&startInitTimer, NULL);
  memset(Y,0,N*M*sizeof(dataType));
  for(int i = 0; i < N; ++i)
      for(int j = 0; j < M; ++j)
          X[i*M + j] = rand1 * (i+1);
  gettimeofday(&endInitTimer, NULL);

  gettimeofday(&startMemCpyTimer, NULL);
  checkCudaErrors(cudaMemcpy(d_X, X, N*M*sizeof(dataType), cudaMemcpyHostToDevice));
  gettimeofday(&endMemCpyTimer, NULL);
  elapsed_memcpy += elapsedTime(startMemCpyTimer, endMemCpyTimer);

#if !defined(VERIFY_GPU_CORRECTNESS)
  //Run the kernel couple of times before the actual timings begins
  gettimeofday(&startInitKernel, NULL);
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  checkCudaErrors(cudaDeviceSynchronize());
  gettimeofday(&endInitKernel, NULL);
#endif

  //Start Kernel Timer
  gettimeofday(&startKernelTimer, NULL);

  //Start actual kernel//Start Kernel Timerpinned
  for(int iter = 0; iter < run_kernel_times; ++iter)
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  checkCudaErrors(cudaDeviceSynchronize());

  //End Kernel Timer
  gettimeofday(&endKernelTimer, NULL);

  gettimeofday(&startMemCpyTimer, NULL);
  checkCudaErrors(cudaMemcpy(Y, d_Y, N*M*sizeof(dataType), cudaMemcpyDeviceToHost));
  gettimeofday(&endMemCpyTimer, NULL);

#if VERIFY_GPU_CORRECTNESS
  cout << "PINNED-MEMORY : \t" ;
  checkGPUCorrectness(N,M,Y,yOrig);
#endif

#if !defined(ON_SUMMIT)
  checkCudaErrors(cudaFreeHost(X));
  checkCudaErrors(cudaFreeHost(Y));
#endif
  checkCudaErrors(cudaFree(d_X));
  checkCudaErrors(cudaFree(d_Y));

  gettimeofday(&endTotalTimer, NULL);

  elapsed_memAlloc += elapsedTime(startMemAllocTimer, endMemAllocTimer);
  elapsed_init += elapsedTime(startInitTimer, endInitTimer);
  elapsed_memcpy += elapsedTime(startMemCpyTimer, endMemCpyTimer);
  elapsed_kernel += elapsedTime(startKernelTimer, endKernelTimer);
  elapsed_total += elapsedTime(startTotalTimer, endTotalTimer);
  elapsed_init_kernel += elapsedTime(startInitKernel, endInitKernel);
  nvtxRangePop();
}

void pageable_host_device_memory(dataType a, dataType rand1, double &elapsed_memAlloc, double& elapsed_memcpy, double& elapsed_init, double& elapsed_init_kernel, double &elapsed_kernel, double& elapsed_total, int N, int M, dataType* yOrig)
{
  nvtxRangePushA("pageable_memory");
  int device;
  checkCudaErrors(cudaSetDevice(0));
  checkCudaErrors(cudaGetDevice(&device));
  timeval startMemAllocTimer, endMemAllocTimer,
          startInitTimer, endInitTimer,
          startKernelTimer, endKernelTimer, 
          startTotalTimer, endTotalTimer, 
          startInitKernel, endInitKernel, 
          startMemCpyTimer, endMemCpyTimer;

  gettimeofday(&startTotalTimer, NULL);

  dataType *Y, *X;
  dataType *d_Y, *d_X;
  dim3 grid(N,1,1);
  dim3 threads(32,1,1);

  gettimeofday(&startMemAllocTimer, NULL);
  X = (dataType*) malloc(N*M*sizeof(dataType));
  Y = (dataType*) malloc(N*M*sizeof(dataType));

  //Allocate memory on device
  checkCudaErrors(cudaMalloc(&d_X, N*M*sizeof(dataType)));
  checkCudaErrors(cudaMalloc(&d_Y, N*M*sizeof(dataType)));
  gettimeofday(&endMemAllocTimer, NULL);

  gettimeofday(&startInitTimer, NULL);
  memset(Y,0,N*M*sizeof(dataType));
  for(int i = 0; i < N; ++i)
      for(int j = 0; j < M; ++j)
          X[i*M + j] = rand1 * (i+1);
  gettimeofday(&endInitTimer, NULL);

  gettimeofday(&startMemCpyTimer, NULL);
  checkCudaErrors(cudaMemcpy(d_X, X, N*M*sizeof(dataType), cudaMemcpyHostToDevice));
  gettimeofday(&endMemCpyTimer, NULL);
  elapsed_memcpy += elapsedTime(startMemCpyTimer, endMemCpyTimer);

  //Run the kernel couple of times before the actual timings begins
#if !defined(VERIFY_GPU_CORRECTNESS)
  gettimeofday(&startInitKernel, NULL);
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  checkCudaErrors(cudaDeviceSynchronize());
  gettimeofday(&endInitKernel, NULL);
#endif

  //Start Kernel Timer
  gettimeofday(&startKernelTimer, NULL);

  //Start actual kernel//Start Kernel Timerpinned
  for(int iter = 0; iter < run_kernel_times; ++iter)
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  checkCudaErrors(cudaDeviceSynchronize());

  //End Kernel Timer
  gettimeofday(&endKernelTimer, NULL);

  gettimeofday(&startMemCpyTimer, NULL);
  checkCudaErrors(cudaMemcpy(Y, d_Y, N*M*sizeof(dataType), cudaMemcpyDeviceToHost));
  gettimeofday(&endMemCpyTimer, NULL);
  elapsed_memcpy += elapsedTime(startMemCpyTimer, endMemCpyTimer);

#if VERIFY_GPU_CORRECTNESS
  cout << "PAGEABLE-MEMORY : \t" ;
  checkGPUCorrectness(N,M,Y,yOrig);
#endif

  gettimeofday(&endTotalTimer, NULL);
  free(X);
  free(Y);
  checkCudaErrors(cudaFree(d_X));
  checkCudaErrors(cudaFree(d_Y));

  elapsed_total += elapsedTime(startTotalTimer, endTotalTimer);
  elapsed_memAlloc += elapsedTime(startMemAllocTimer, endMemAllocTimer);
  elapsed_init += elapsedTime(startInitTimer, endInitTimer);
  elapsed_memcpy += elapsedTime(startMemCpyTimer, endMemCpyTimer);
  elapsed_kernel += elapsedTime(startKernelTimer, endKernelTimer);
  elapsed_init_kernel += elapsedTime(startInitKernel, endInitKernel);
  nvtxRangePop();
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
      else if(argc == 4)
      {
          N = atoi(argv[1]);
          M = atoi(argv[2]);
          print_csv = true;
      }
      else
      {
          cout << "Input format is ./axpy.ex N M \n" << endl;
          exit(EXIT_FAILURE);
      }
  }

  timeval startTotalTimer, endTotalTimer;
  gettimeofday(&startTotalTimer, NULL);

  fprintf(stdout, "M = %d\t N = %d\n", M,N);
  fprintf(stdout, "Total Memory Footprint = %f GBs\n", (double)(M*N*sizeof(dataType)/(1024.0*1024.0*1024.0)));
  fprintf(stdout, "threadblocks = %d\t and data accessed by each threadblock = %f Kb\n",N,(double)(M*sizeof(double)/1024.0));

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

  dataType a = 0.5;
  dataType rand1 = (dataType)rand() / (dataType)RAND_MAX;

  dataType *yOrig, *X;
#if VERIFY_GPU_CORRECTNESS
  yOrig = (dataType*) malloc(check_num_values*M*sizeof(dataType));
  X = (dataType*) malloc(check_num_values*M*sizeof(dataType));
  PreComputeFewValues(N,M,a,rand1,yOrig,X);
#endif


  //Allocating data
#if defined(USE_HOST_PAGEABLE_AND_DEVICE_MEMORY)
  double elapsed_memAlloc, elapsed_kernel, elapsed_init, elapsed_memcpy;
  printf("###############Using HOST_PAGEABLE_AND_DEVICE_MEMORY###############\n");
  pageable_host_device_memory(a,rand1,elapsed_memAlloc, elapsed_memcpy, elapsed_init, elapsed_kernel,N,M,yOrig);

#elif defined(USE_PINNED_MEMORY)
  double elapsed_memAlloc, elapsed_kernel, elapsed_init, elapsed_memcpy;
  printf("###############Using PINNED_MEMORY###############\n");
  pinned_memory(a,rand1,elapsed_memAlloc, elapsed_memcpy, elapsed_init, elapsed_kernel,N,M,yOrig);

#elif defined(USE_MANAGED_MEMORY)
  double elapsed_memAlloc, elapsed_kernel, elapsed_init, elapsed_memcpy;
  printf("###############Using MANAGED_MEMORY###############\n");
  managed_memory(a,rand1,elapsed_memAlloc, elapsed_memcpy, elapsed_init, elapsed_kernel,N,M,yOrig);

#elif defined(USE_ZERO_COPY)
  double elapsed_memAlloc, elapsed_kernel, elapsed_init, elapsed_memcpy;
  printf("###############Using ZERO_COPY###############\n");
  zero_copy(a,rand1,elapsed_memAlloc, elapsed_memcpy, elapsed_init, elapsed_kernel,N,M,yOrig);

  //Run all the kernels
#elif defined(RUN_ALL)
  fprintf(stdout,"###############Running All kernels T[sec] ###############\n");
  double pageable_elapsed_memAlloc = 0.0, pageable_elapsed_kernel = 0.0, pageable_init, pageable_memcpy = 0.0, pageable_total = 0.0, pageable_init_kernel = 0.0,
         managed_elapsed_memAlloc = 0.0, managed_elapsed_kernel = 0.0, managed_init = 0.0, managed_memcpy = 0.0, managed_total = 0.0, managed_init_kernel = 0.0,
         pinned_elapsed_memAlloc = 0.0, pinned_elapsed_kernel = 0.0, pinned_init = 0.0, pinned_memcpy = 0.0, pinned_total = 0.0, pinned_init_kernel = 0.0,
         zero_elapsed_memAlloc = 0.0, zero_elapsed_kernel = 0.0, zero_init = 0.0, zero_memcpy = 0.0, zero_total = 0.0, zero_init_kernel = 0.0;
#if defined(ON_SUMMIT)
  double tUVM_elapsed_memAlloc = 0.0, tUVM_elapsed_kernel = 0.0, tUVM_init = 0.0, tUVM_memcpy = 0.0, tUVM_total = 0.0, tUVM_init_kernel = 0.0;
#endif

#if !defined(VERIFY_GPU_CORRECTNESS)
  //Run the job for NUM_LOOPS number of times 
  for(int iter = 0; iter < NUM_LOOPS; ++iter) 
#endif
  {
    pageable_host_device_memory(a, rand1, pageable_elapsed_memAlloc, pageable_memcpy, pageable_init, pageable_init_kernel, pageable_elapsed_kernel, pageable_total, N, M, yOrig);
    pinned_memory(a, rand1, pinned_elapsed_memAlloc, pinned_memcpy, pinned_init, pinned_init_kernel, pinned_elapsed_kernel, pinned_total, N, M, yOrig);
    managed_memory(a, rand1, managed_elapsed_memAlloc, managed_memcpy, managed_init, managed_init_kernel, managed_elapsed_kernel, managed_total, N, M, yOrig);
    zero_copy(a, rand1, zero_elapsed_memAlloc, zero_memcpy, zero_init, zero_init_kernel, zero_elapsed_kernel, zero_total, N, M, yOrig);

    //If on summit, run the true-UVM kernel
#if defined(ON_SUMMIT)
    tUVM(a, rand1, tUVM_elapsed_memAlloc, tUVM_memcpy, tUVM_init, tUVM_init_kernel, tUVM_elapsed_kernel, tUVM_total, N, M, yOrig);
#endif
  }

#if !defined(VERIFY_GPU_CORRECTNESS)
  //Take the average time for each of the runs
    pageable_elapsed_memAlloc /= NUM_LOOPS; pageable_elapsed_kernel /= NUM_LOOPS; pageable_init /= NUM_LOOPS; pageable_memcpy /= NUM_LOOPS; pageable_total /= NUM_LOOPS; pageable_init_kernel /= NUM_LOOPS;
    managed_elapsed_memAlloc /= NUM_LOOPS; managed_elapsed_kernel /= NUM_LOOPS; managed_init /= NUM_LOOPS; managed_memcpy /= NUM_LOOPS; managed_total /= NUM_LOOPS; managed_init_kernel /= NUM_LOOPS;
    pinned_elapsed_memAlloc /= NUM_LOOPS; pinned_elapsed_kernel /= NUM_LOOPS; pinned_init /= NUM_LOOPS; pinned_memcpy /= NUM_LOOPS; pinned_total /= NUM_LOOPS; pinned_init_kernel /= NUM_LOOPS;
    zero_elapsed_memAlloc /= NUM_LOOPS; zero_elapsed_kernel /= NUM_LOOPS; zero_init /= NUM_LOOPS; zero_memcpy /= NUM_LOOPS, zero_total /= NUM_LOOPS; zero_init_kernel /= NUM_LOOPS;
#if (ON_SUMMIT)
    tUVM_elapsed_memAlloc /= NUM_LOOPS; tUVM_elapsed_kernel /= NUM_LOOPS; tUVM_init /= NUM_LOOPS; tUVM_memcpy /= NUM_LOOPS; tUVM_init_kernel /= NUM_LOOPS;
#endif
#endif

  cudaDeviceSynchronize();
#endif

#if USE_TIMEMORY
  data_allocation.stop();
#endif

#if USE_TIMEMORY
    measure.stop();
#endif
  gettimeofday(&endTotalTimer, NULL);

  //calculate elapsed time
  double elapsed_total = elapsedTime(startTotalTimer, endTotalTimer);

#if RUN_ALL
  if(print_csv)
  {
    fprintf(stdout, "Device, Memory-Type, MemAlloc-time, MemCPY-time, Initial-Kernel-time, Kernel-time, Init-Values, Kernel+total,\n");
    fprintf(stdout, "0, pageable, %f, %f, %f, %f, %f, %f,\n", pageable_elapsed_memAlloc, pageable_memcpy, pageable_init_kernel, pageable_elapsed_kernel, pageable_init, pageable_total);
    fprintf(stdout, "1, host-pinned, %f, %f, %f, %f, %f, %f,\n", pinned_elapsed_memAlloc, pinned_memcpy, pinned_init_kernel, pinned_elapsed_kernel, pinned_init,pinned_total);
    fprintf(stdout, "2, managed, %f, %f, %f, %f, %f, %f,\n", managed_elapsed_memAlloc, managed_memcpy, managed_init_kernel, managed_elapsed_kernel, managed_init, managed_total);
    fprintf(stdout, "3, zero-copy, %f, %f, %f, %f, %f, %f,\n", zero_elapsed_memAlloc, zero_memcpy, zero_init_kernel, zero_elapsed_kernel, zero_init, zero_total);
#if (ON_SUMMIT)
    fprintf(stdout, "4, true-UVM, %f, %f, %f, %f, %f, %f,\n", tUVM_elapsed_memAlloc, tUVM_memcpy, tUVM_init_kernel, tUVM_elapsed_kernel, tUVM_init, tUVM_total);
#endif
  }
  else
  {
    fprintf(stdout, "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    fprintf(stdout, "Device \t Memory-Type \t MemAlloc-time \t\t MemCPY-time \t\t Initial-Kernel-time \t\t Kernel-time \t\t Init-Values \t\t Kernel+total\n");
    fprintf(stdout, "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    fprintf(stdout, "0 \t pageable \t %f \t\t %f \t\t %f \t\t\t %f \t\t %f \t\t %f\n", pageable_elapsed_memAlloc, pageable_memcpy, pageable_init_kernel, pageable_elapsed_kernel, pageable_init, pageable_total);
    fprintf(stdout, "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    fprintf(stdout, "1 \t host-pinned \t %f \t\t %f \t\t %f \t\t\t %f \t\t %f \t\t %f\n", pinned_elapsed_memAlloc, pinned_memcpy, pinned_init_kernel, pinned_elapsed_kernel, pinned_init,pinned_total);
    fprintf(stdout, "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    fprintf(stdout, "2 \t managed \t %f \t\t %f \t\t %f \t\t\t %f \t\t %f \t\t %f\n", managed_elapsed_memAlloc, managed_memcpy, managed_init_kernel, managed_elapsed_kernel, managed_init, managed_total);
    fprintf(stdout, "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
    fprintf(stdout, "3 \t zero-copy \t %f \t\t %f \t\t %f \t\t\t %f \t\t %f \t\t %f\n", zero_elapsed_memAlloc, zero_memcpy, zero_init_kernel, zero_elapsed_kernel, zero_init, zero_total);
    fprintf(stdout, "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
#if (ON_SUMMIT)
    fprintf(stdout, "4 \t true-UVM \t %f \t\t %f \t\t %f \t\t\t %f \t\t %f \t\t %f\n", tUVM_elapsed_memAlloc, tUVM_memcpy, tUVM_init_kernel, tUVM_elapsed_kernel, tUVM_init, tUVM_total);
    fprintf(stdout, "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
#endif
  }

  fprintf(stdout, "Total time = %f [sec]\n", elapsed_total);
  fprintf(stdout, "###################################################################################################\n");

#else
  cout << "************ MemAlloc-time = " << elapsed_memAlloc << " [sec] ************\n" << endl;
  cout << "************ Kernel-time = " << elapsed_kernel << " [sec] ************\n" << endl;
  cout << "************ Total-time = " << elapsed_total << " [sec] ************\n" << endl;
#endif
    return 0;
}
