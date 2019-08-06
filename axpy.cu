
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
#define NUM_LOOPS 10
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
    for(int i = blockIdx.x; i < N; i+=gridDim.x)
        for(int j = threadIdx.x; j < M; j+=blockDim.x)
            Y(i,j) += a * X(i,j);
}

void zero_copy(dataType a, dataType rand1, double& elapsed_memAlloc, double& elapsed_memcpy, double& elapsed_init, double& elapsed_kernel, int N, int M, dataType* yOrig)
{
  nvtxRangePushA("Zero_copy");
  int device;
  checkCudaErrors(cudaSetDevice(3));
  checkCudaErrors(cudaGetDevice(&device));
  timeval startMemAllocTimer, endMemAllocTimer,
          startInitTimer, endInitTimer,
          startKernelTimer, endKernelTimer;

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
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  checkCudaErrors(cudaDeviceSynchronize());
#endif

  //Start Kernel Timer
  gettimeofday(&startKernelTimer, NULL);

  //Start actual kernel//Start Kernel Timerpinned
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

  elapsed_memAlloc += elapsedTime(startMemAllocTimer, endMemAllocTimer);
  elapsed_init += elapsedTime(startInitTimer, endInitTimer);
  elapsed_kernel += elapsedTime(startKernelTimer, endKernelTimer);
  nvtxRangePop();
}


void managed_memory(dataType a, dataType rand1, double &elapsed_memAlloc, double& elapsed_memcpy, double& elapsed_init, double &elapsed_kernel, int N, int M, dataType *yOrig)
{
  nvtxRangePushA("managed_memory");
  int device;
  checkCudaErrors(cudaSetDevice(2));
  checkCudaErrors(cudaGetDevice(&device));
  timeval startMemAllocTimer, endMemAllocTimer,
          startInitTimer, endInitTimer,
          startKernelTimer, endKernelTimer;

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
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  checkCudaErrors(cudaDeviceSynchronize());
#endif

  //Start Kernel Timer
  gettimeofday(&startKernelTimer, NULL);

  //Start actual kernel//Start Kernel Timerpinned
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

  elapsed_memAlloc += elapsedTime(startMemAllocTimer, endMemAllocTimer);
  elapsed_init += elapsedTime(startInitTimer, endInitTimer);
  elapsed_kernel += elapsedTime(startKernelTimer, endKernelTimer);
  nvtxRangePop();
}

void pinned_memory(dataType a, dataType rand1, double &elapsed_memAlloc, double& elapsed_memcpy, double& elapsed_init, double &elapsed_kernel, int N, int M, dataType* yOrig)
{
  nvtxRangePushA("pinned_memory");
  int device;
  checkCudaErrors(cudaSetDevice(1));
  checkCudaErrors(cudaGetDevice(&device));
  timeval startMemAllocTimer, endMemAllocTimer,
          startInitTimer, endInitTimer,
          startKernelTimer, endKernelTimer, 
          startMemCpyTimer, endMemCpyTimer;

  dataType *Y, *X;
  dataType *d_Y, *d_X;
  dim3 grid(N,1,1);
  dim3 threads(32,1,1);

  gettimeofday(&startMemAllocTimer, NULL);
//  checkCudaErrors(cudaHostAlloc(&X, N*M*sizeof(dataType), cudaHostAllocDefault));
//  checkCudaErrors(cudaHostAlloc(&Y, N*M*sizeof(dataType), cudaHostAllocDefault));
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
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  checkCudaErrors(cudaDeviceSynchronize());
#endif

  //Start Kernel Timer
  gettimeofday(&startKernelTimer, NULL);

  //Start actual kernel//Start Kernel Timerpinned
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

  elapsed_memAlloc += elapsedTime(startMemAllocTimer, endMemAllocTimer);
  elapsed_init += elapsedTime(startInitTimer, endInitTimer);
  elapsed_memcpy += elapsedTime(startMemCpyTimer, endMemCpyTimer);
  elapsed_kernel += elapsedTime(startKernelTimer, endKernelTimer);
  nvtxRangePop();
}

void pageable_host_device_memory(dataType a, dataType rand1, double &elapsed_memAlloc, double& elapsed_memcpy, double& elapsed_init, double &elapsed_kernel, int N, int M, dataType* yOrig)
{
  nvtxRangePushA("pageable_memory");
  int device;
  checkCudaErrors(cudaSetDevice(0));
  checkCudaErrors(cudaGetDevice(&device));
  timeval startMemAllocTimer, endMemAllocTimer,
          startInitTimer, endInitTimer,
          startKernelTimer, endKernelTimer, 
          startMemCpyTimer, endMemCpyTimer;

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
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  checkCudaErrors(cudaDeviceSynchronize());
#endif

  //Start Kernel Timer
  gettimeofday(&startKernelTimer, NULL);

  //Start actual kernel//Start Kernel Timerpinned
  axpyKernel <<<grid,threads>>> (N,M,a,d_Y,d_X);
  checkCudaErrors(cudaDeviceSynchronize());

  //End Kernel Timer
  gettimeofday(&endKernelTimer, NULL);

  gettimeofday(&startMemCpyTimer, NULL);
  checkCudaErrors(cudaMemcpy(Y, d_Y, N*M*sizeof(dataType), cudaMemcpyDeviceToHost));
  gettimeofday(&endMemCpyTimer, NULL);
  elapsed_memcpy += elapsedTime(startMemCpyTimer, endMemCpyTimer);

  elapsed_memAlloc += elapsedTime(startMemAllocTimer, endMemAllocTimer);
  elapsed_init += elapsedTime(startInitTimer, endInitTimer);
  elapsed_memcpy += elapsedTime(startMemCpyTimer, endMemCpyTimer);
  elapsed_kernel += elapsedTime(startKernelTimer, endKernelTimer);

#if VERIFY_GPU_CORRECTNESS
  cout << "PAGEABLE-MEMORY : \t" ;
  checkGPUCorrectness(N,M,Y,yOrig);
#endif

  free(X);
  free(Y);
  checkCudaErrors(cudaFree(d_X));
  checkCudaErrors(cudaFree(d_Y));
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

  cout << "M = " << M << "\t N = " << N << endl;
  cout << "Total Memory Footprint = " << (double)(M*N*sizeof(dataType)/(1024.0*1024.0*1024.0)) << " GBs" << endl;
  cout << "threadblocks = " << N << "  and data accessed by each threadblock = " << (double)(M*sizeof(double)/1024.0) << " Kb" << endl;

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
  double elapsed_memAlloc, elapsed_kernel, elapsed_init;
  printf("###############Using HOST_PAGEABLE_AND_DEVICE_MEMORY###############\n");
  pageable_host_device_memory(a,rand1,elapsed_memAlloc,elapsed_init, elapsed_kernel,N,M,yOrig);

#elif defined(USE_PINNED_MEMORY)
  double elapsed_memAlloc, elapsed_kernel, elapsed_init;
  printf("###############Using PINNED_MEMORY###############\n");
  pinned_memory(a,rand1,elapsed_memAlloc,elapsed_init,elapsed_kernel,N,M,yOrig);

#elif defined(USE_MANAGED_MEMORY)
  double elapsed_memAlloc, elapsed_kernel, elapsed_init;
  printf("###############Using MANAGED_MEMORY###############\n");
  managed_memory(a,rand1,elapsed_memAlloc,elapsed_init,elapsed_kernel,N,M,yOrig);

#elif defined(USE_ZERO_COPY)
  double elapsed_memAlloc, elapsed_kernel, elapsed_init;
  printf("###############Using ZERO_COPY###############\n");
  zero_copy(a,rand1,elapsed_memAlloc,elapsed_init,elapsed_kernel,N,M,yOrig);

  //Run all the kernels
#elif defined(RUN_ALL)
  printf("###############Running All kernels###############\n");
  double pageable_elapsed_memAlloc = 0.0, pageable_elapsed_kernel = 0.0, pageable_init, pageable_memcpy = 0.0,
         managed_elapsed_memAlloc = 0.0, managed_elapsed_kernel = 0.0, managed_init = 0.0, managed_memcpy = 0.0,
         pinned_elapsed_memAlloc = 0.0, pinned_elapsed_kernel = 0.0, pinned_init = 0.0, pinned_memcpy = 0.0,
         zero_elapsed_memAlloc = 0.0, zero_elapsed_kernel = 0.0, zero_init = 0.0, zero_memcpy = 0.0;

#if !defined(VERIFY_GPU_CORRECTNESS)
  //Run the job for NUM_LOOPS number of times 
  for(int iter = 0; iter < NUM_LOOPS; ++iter) 
#endif
  {
    pageable_host_device_memory(a, rand1, pageable_elapsed_memAlloc, pageable_memcpy, pageable_init, pageable_elapsed_kernel, N, M, yOrig);
    pinned_memory(a, rand1, pinned_elapsed_memAlloc, pinned_memcpy, pinned_init, pinned_elapsed_kernel, N, M, yOrig);
    managed_memory(a, rand1, managed_elapsed_memAlloc, managed_memcpy, managed_init, managed_elapsed_kernel, N, M, yOrig);
    zero_copy(a, rand1, zero_elapsed_memAlloc, zero_memcpy, zero_init, zero_elapsed_kernel, N, M, yOrig);
  }

#if !defined(VERIFY_GPU_CORRECTNESS)
  //Take the average time for each of the runs
    pageable_elapsed_memAlloc /= NUM_LOOPS; pageable_elapsed_kernel /= NUM_LOOPS; pageable_init /= NUM_LOOPS; pageable_memcpy /= NUM_LOOPS;
    managed_elapsed_memAlloc /= NUM_LOOPS; managed_elapsed_kernel /= NUM_LOOPS; managed_init /= NUM_LOOPS; managed_memcpy /= NUM_LOOPS;
    pinned_elapsed_memAlloc /= NUM_LOOPS; pinned_elapsed_kernel /= NUM_LOOPS; pinned_init /= NUM_LOOPS; pinned_memcpy /= NUM_LOOPS;
    zero_elapsed_memAlloc /= NUM_LOOPS; zero_elapsed_kernel /= NUM_LOOPS; zero_init /= NUM_LOOPS; zero_memcpy /= NUM_LOOPS;
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
    fprintf(stderr, "Device, \t Memory-Type, \t MemAlloc-time[sec], \t MemCPY-time[sec], \t Kernel-time[sec], \t Kernel+MemAlloc[sec], \t Init-Values[sec]\n");
    fprintf(stderr, "0, \t pageable, \t %f, \t\t %f, \t\t %f, \t\t %f, \t\t %f\n", pageable_elapsed_memAlloc, pageable_memcpy, pageable_elapsed_kernel, pageable_elapsed_memAlloc+pageable_elapsed_kernel, pageable_init);
    fprintf(stderr, "1, \t host-pinned, \t %f, \t\t %f, \t\t %f, \t\t %f, \t\t %f, \n", pinned_elapsed_memAlloc, pinned_memcpy, pinned_elapsed_kernel, pinned_elapsed_memAlloc+pinned_elapsed_kernel,pinned_init);
    fprintf(stderr, "2, \t managed, \t %f, \t\t %f, \t\t %f, \t\t %f, \t\t %f, \n", managed_elapsed_memAlloc, managed_memcpy, managed_elapsed_kernel, managed_elapsed_memAlloc+managed_elapsed_kernel, managed_init);
    fprintf(stderr, "3, \t zero-copy, \t %f, \t\t %f, \t\t %f, \t\t %f, \t\t %f \n", zero_elapsed_memAlloc, zero_memcpy, zero_elapsed_kernel, zero_elapsed_memAlloc+zero_elapsed_kernel, zero_init);
  }
  else
  {
    fprintf(stderr, "-------------------------------------------------------------------------------------------------------------------------------------------\n");
    fprintf(stderr, "Device, \t Memory-Type, \t MemAlloc-time[sec], \t MemCPY-time[sec], \t Kernel-time[sec], \t Kernel+MemAlloc[sec], \t Init-Values[sec]\n");
    fprintf(stderr, "-------------------------------------------------------------------------------------------------------------------------------------------\n");
    fprintf(stderr, "0 \t pageable, \t %f, \t\t %f, \t\t %f, \t\t %f, \t\t %f\n", pageable_elapsed_memAlloc, pageable_memcpy, pageable_elapsed_kernel, pageable_elapsed_memAlloc+pageable_elapsed_kernel, pageable_init);
    fprintf(stderr, "-------------------------------------------------------------------------------------------------------------------------------------------\n");
    fprintf(stderr, "1 \t host-pinned, \t %f, \t\t %f, \t\t %f, \t\t %f, \t\t %f, \n", pinned_elapsed_memAlloc, pinned_memcpy, pinned_elapsed_kernel, pinned_elapsed_memAlloc+pinned_elapsed_kernel,pinned_init);
    fprintf(stderr, "-------------------------------------------------------------------------------------------------------------------------------------------\n");
    fprintf(stderr, "2 \t managed, \t %f, \t\t %f, \t\t %f, \t\t %f, \t\t %f, \n", managed_elapsed_memAlloc, managed_memcpy, managed_elapsed_kernel, managed_elapsed_memAlloc+managed_elapsed_kernel, managed_init);
    fprintf(stderr, "-------------------------------------------------------------------------------------------------------------------------------------------\n");
    fprintf(stderr, "3 \t zero-copy, \t %f, \t\t %f, \t\t %f, \t\t %f, \t\t %f \n", zero_elapsed_memAlloc, zero_memcpy, zero_elapsed_kernel, zero_elapsed_memAlloc+zero_elapsed_kernel, zero_init);
    fprintf(stderr, "-------------------------------------------------------------------------------------------------------------------------------------------\n");
  }

  cout << "************ Total-time = " << elapsed_total << " [sec] ************\n" << endl;

#else
  cout << "************ MemAlloc-time = " << elapsed_memAlloc << " [sec] ************\n" << endl;
  cout << "************ Kernel-time = " << elapsed_kernel << " [sec] ************\n" << endl;
  cout << "************ Total-time = " << elapsed_total << " [sec] ************\n" << endl;
#endif
    return 0;
}
