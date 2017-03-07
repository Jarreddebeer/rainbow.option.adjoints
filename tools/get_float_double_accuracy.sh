#!/bin/bash

run_simulation() {
    ACCURACY_FILE=$2
    BIN=$3
    INFILE=$4

    for ((simSize=5120;simSize<=501760;simSize+=5120));
    do
        echo $simSize
        echo $simSize 360 $(./tools/run.sh $BIN $simSize 360 $INFILE) > tmp.txt
        cat tmp.txt | head -n 9 | tail -n 7 | head -n 1 | awk '{print $3}' >> $ACCURACY_FILE # price
        cat tmp.txt | head -n 9 | tail -n 6 | head -n 1 | awk '{print $3}' >> $ACCURACY_FILE # delta 0
        cat tmp.txt | head -n 9 | tail -n 5 | head -n 1 | awk '{print $3}' >> $ACCURACY_FILE # vega  0
        cat tmp.txt | head -n 9 | tail -n 4 | head -n 1 | awk '{print $3}' >> $ACCURACY_FILE # delta 1
        cat tmp.txt | head -n 9 | tail -n 3 | head -n 1 | awk '{print $3}' >> $ACCURACY_FILE # vega  1
        cat tmp.txt | head -n 9 | tail -n 2 | head -n 1 | awk '{print $3}' >> $ACCURACY_FILE # delta 2
        cat tmp.txt | head -n 9 | tail -n 1 | head -n 1 | awk '{print $3}' >> $ACCURACY_FILE # vega  2
    done
}
