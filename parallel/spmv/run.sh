#!/bin/bash

for t in {1.."$1"}
do
	echo "Running run_spmv.jl with $t threads"
	julia --threads="$t" "run_spmv.jl"
done
