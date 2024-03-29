# GPUMemoryTest
### Makefile build : 

```shell
make [options]
  PINNED_MEMORY - Use pinned memory
  MANAGED_MEMORY - Use managed memory
  ZERO_COPY - Use zero copy
  TIMEMORY_PROFILE - use TiMemory to profile the kernels
  Default - Runs all kernels and prints out the timing
  CHECK_CORRECTNESS - To verify GPU correctness
```

```shell
  For prefetch : 
  make tUVM_prefetch=y managed_prefetch=y 
```


### CMAKE build : 

``` shell
cmake [options] : 
  PINNED_MEMORY - Use pinned memory
  MANAGED_MEMORY - Use managed memory
  ZERO_COPY - Use zero copy
  Default - HOST pageable and Deveice memory
  TIMEMORY_PROFILE - use TiMemory to profile the kernels
```


### Run the application :

```shell
  ./axpy.ex N M inner outer
```

### On Summit : 

```shell
jsrun -n 1 -a 1 -c 1 -g 6 ./axpy.ex N M inner outer
```

### To redirect the numbers to a csv-file

```shell
outer - number of rows
inner - number of collumns
1 - Prepare the output for printing it out to csv
1 - redirect stdout output
2 - redirect sterr output
./axpy.ex N M 1 1>> sheet.csv 2>> sheet.csv
```
