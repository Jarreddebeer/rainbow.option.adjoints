#!/bin/bash

BIN=$1
SIM_SIZE=$2
STEP_SIZE=$3
INFILE=$4

get_time() {
    $1 $2 $3 | tail -n 2 | head -n 1 | awk '{print $2}'
}

echo $BIN $SIM_SIZE $STEP_SIZE $ASSETS
get_time $BIN $SIM_SIZE $STEP_SIZE < $INFILE
get_time $BIN $SIM_SIZE $STEP_SIZE < $INFILE
get_time $BIN $SIM_SIZE $STEP_SIZE < $INFILE
get_time $BIN $SIM_SIZE $STEP_SIZE < $INFILE
get_time $BIN $SIM_SIZE $STEP_SIZE < $INFILE

echo ''
