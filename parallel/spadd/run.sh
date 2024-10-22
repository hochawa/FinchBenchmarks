#!/bin/bash

for (( t=1 ; t<=$1 ; t++));
do
    echo "Running run_spadd.jl with $t threads"
    julia --threads="$t" "run_spadd.jl" -a
done
