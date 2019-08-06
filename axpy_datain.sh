#!/bin/bash
#run axpy for the following datasets
./axpy.ex 100 128 #1KB threadblock

for i in {1,10,100,1000,10000,10000,100000}
do
  jsrun -n 1 -a 1 -c 1 -g 6 ./axpy.ex 100 $[i*100] 1 1>> summit_perf.csv 2>> summit_perf.csv
done
