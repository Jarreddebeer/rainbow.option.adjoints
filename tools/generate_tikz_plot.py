#!/bin/python

import sys
from tikz import *

colors = {"OMP": "blue",
          "CUDA00":  "{rgb,255:red,0; green,105; blue,0}",
          "CUDA01":  "{rgb,255:red,0; green,135; blue,0}",
          "CUDA02":  "{rgb,255:red,0; green,165; blue,0}",
          "CUDA03a": "{rgb,255:red,0; green,195; blue,0}",
          "CUDA03b": "{rgb,255:red,0; green,225; blue,0}",
          "CUDA04":  "{rgb,255:red,0; green,255; blue,0}",
          "CUDA06":  "{rgb,255:red,0; green,0; blue,255}",
          "CUDA07":  "{rgb,255:red,255; green,0; blue,0}",
}

symbols = {"OMP": "x",
          "CUDA00":  "*",
          "CUDA01":  "*",
          "CUDA02":  "*",
          "CUDA03a": "*",
          "CUDA03b": "*",
          "CUDA04":  "*",
          "CUDA06": "*",
          "CUDA07": "*"
}

usedKeys = []
data = {}
dataDict = {}

# extract lines

for line in sys.stdin:

    line = line[:-1]

    if len(line) >= 3 and len(line) <= 7:
        if line != "SERIAL":
            usedKeys.append(line)
        if line not in data:
            data[line] = {}
        dataDict = data[line]
        continue

    simSize, stepSize, timeTaken = map(float, line.split(" "))

    if stepSize not in dataDict:
        dataDict[stepSize] = {}

    if simSize not in dataDict[stepSize]:
        dataDict[stepSize][simSize] = {}

    dataDict[stepSize][simSize] = timeTaken

# convert time taken to speedup and generate plot

for stepSize in data["SERIAL"]:

    picture = TikzPicture ('Speedup', 'Speedup at ' + str(stepSize) + ' time steps', 'Number of paths', "Acceleration")

    for runtype in usedKeys:
        plot = TikzPlot(runtype + " " + str(int(stepSize)),  colors[runtype],  symbols[runtype])
        for simSize in data["SERIAL"][stepSize]:
            cpuTime  = data["SERIAL"][stepSize][simSize]
            speedup = cpuTime / data[runtype][stepSize][simSize]
            plot.add_point(int(simSize), speedup)
        picture.add_plot(plot)

    picture.generate()

