# GPUMemoryTest
Makefile build : 
make [options]
  PINNED_MEMORY - Use pinned memory
  MANAGED_MEMORY - Use managed memory
  ZERO_COPY - Use zero copy
  TIMEMORY_PROFILE - use TiMemory to profile the kernels
  Default - Runs all kernels and prints out the timing
  CHECK_CORRECTNESS - To verify GPU correctness

CMAKE build : 
cmake [options] : 
  PINNED_MEMORY - Use pinned memory
  MANAGED_MEMORY - Use managed memory
  ZERO_COPY - Use zero copy
  Default - HOST pageable and Deveice memory
  TIMEMORY_PROFILE - use TiMemory to profile the kernels

Run the application : 
  ./axpy.ex N M 

  To redirect the numbers to a csv-file
  N - number of rows
  M - number of collumns
  1 - Prepare the output for printing it out to csv
  1 - redirect stdout output
  2 - redirect sterr output
  ./axpy.ex N M 1 1>> sheet.csv 2>> sheet.csv
