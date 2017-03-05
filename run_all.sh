#!/bin/bash

run_simulations() {
    METHOD=$1
    TIMING_FILE=$2
    BIN=$3
    INFILE=$4

    echo $METHOD >> $TIMING_FILE
    for ((simSize=10048;simSize<=30000;simSize+=10048));
    do
        echo $simSize
        echo $simSize 360 $(./tools/run.sh $BIN $simSize 360 $INFILE | python ./tools/get_mean.py) | awk '{print $1 " " $2 " " $6}' >> $TIMING_FILE
    done
}

run_simulations SERIAL timing/singlethreaded        ./bin/serial.adjoint           assets.rainbow.in
run_simulations OMP    timing/multithreaded.omp     ./bin/omp.adjoint              assets.rainbow.in
run_simulations CUDA01 timing/multithreaded.cuda.01 ./bin/cuda.01.adjoint          assets.rainbow.in
run_simulations CUDA02 timing/multithreaded.cuda.02 ./bin/cuda.02.adjoint          assets.rainbow.in
run_simulations CUDA03 timing/multithreaded.cuda.03 ./bin/cuda.03.floats           assets.rainbow.in
run_simulations CUDA04 timing/multithreaded.cuda.04 ./bin/cuda.03.float.constants  assets.rainbow.in
