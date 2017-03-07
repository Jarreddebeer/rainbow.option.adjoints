#!/bin/python

import sys

sys.stdout.write("--------------------\n")
sys.stdout.write( sys.stdin.readline())
sys.stdout.write("--------------------\n")

for line in sys.stdin:

    line = line[:-1]
    simSize, stepSize, timeTaken = map(float, line.split(" "))
    sys.stdout.write("(%s %s) " % (int(simSize), timeTaken))

sys.stdout.write("\n")
sys.stdout.flush()
