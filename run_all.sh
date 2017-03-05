#!/bin/bash

run_simulations() {
    METHOD=$1
    TIMING_FILE=$2
    BIN=$3
    INFILE=$4

    rm $TIMING_FILE
    touch $TIMING_FILE
    echo $METHOD >> $TIMING_FILE
    for ((simSize=10048;simSize<=50000;simSize+=10048));
    do
        echo $simSize
        echo $simSize 360 $(./tools/run.sh $BIN $simSize 360 $INFILE | python ./tools/get_mean.py) | awk '{print $1 " " $2 " " $6}' >> $TIMING_FILE
    done
}

run_simulations SERIAL timing.serial ./bin/serial.adjoint assets.rainbow.in
