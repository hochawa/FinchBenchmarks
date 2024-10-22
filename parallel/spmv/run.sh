#!/bin/bash

for (( t=1 ; t<=$1 ; t++));
do
	echo "Running run_spmv.jl with $t threads"
	julia --threads="$t" "run_spmv.jl"
done
