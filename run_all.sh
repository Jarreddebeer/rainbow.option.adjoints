#!/bin/bash

run_simulations() {
    METHOD=$1
    TIMING_FILE=$2
    BIN=$3
    INFILE=$4

    echo $METHOD >> $TIMING_FILE
#    for ((simSize=5120;simSize<=501760;simSize+=5120));
    for ((simSize=5120;simSize<=501760;simSize+=5120));
    do
        echo $simSize
        echo $simSize 360 $(./tools/run.sh $BIN $simSize 360 $INFILE | python ./tools/get_mean.py) | awk '{print $1 " " $2 " " $6}' >> $TIMING_FILE
    done
}

run_simulations SERIAL timing/singlethreaded        ./bin/serial.adjoint           assets.rainbow.in
run_simulations OMP    timing/multithreaded.omp     ./bin/omp.adjoint              assets.rainbow.in
run_simulations CUDA01 timing/multithreaded.01.cuda ./bin/cuda.01.adjoint          assets.rainbow.in
run_simulations CUDA02 timing/multithreaded.02.cuda ./bin/cuda.02.adjoint          assets.rainbow.in
run_simulations CUDA03 timing/multithreaded.03.cuda ./bin/cuda.03.floats           assets.rainbow.in
run_simulations CUDA04 timing/multithreaded.04.cuda ./bin/cuda.04.float.constants  assets.rainbow.in
run_simulations CUDA05 timing/multithreaded.05.cuda ./bin/cuda.05.shared.memory    assets.rainbow.in
run_simulations CUDA06 timing/multithreaded.06.cuda ./bin/cuda.06.fast.math        assets.rainbow.in
run_simulations CUDA07 timing/multithreaded.07.cuda ./bin/cuda.07.float2           assets.rainbow.in
