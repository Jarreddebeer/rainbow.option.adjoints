cuda08:
	/usr/local/cuda-7.5/bin/nvcc -m64 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50 -lineinfo -Xptxas=-v cuda.08.float4.cu -o bin/cuda.08.float4 -use_fast_math

cuda07: 
	/usr/local/cuda-7.5/bin/nvcc -m64 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50 -lineinfo -Xptxas=-v cuda.07.float2.cu -o bin/cuda.07.float2 -use_fast_math

cuda06: 
	/usr/local/cuda-7.5/bin/nvcc -m64 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50 -lineinfo -Xptxas=-v cuda.06.fast.math.cu -o bin/cuda.06.fast.math -use_fast_math

cuda05: 
	/usr/local/cuda-7.5/bin/nvcc -m64 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50 -lineinfo -Xptxas=-v cuda.05.shared.memory.cu -o bin/cuda.05.shared.memory

cuda04: 
	/usr/local/cuda-7.5/bin/nvcc -m64 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50 -lineinfo -Xptxas=-v cuda.04.float.constants.cu -o bin/cuda.04.float.constants

cuda03: 
	/usr/local/cuda-7.5/bin/nvcc -m64 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50 -lineinfo -Xptxas=-v cuda.03.floats.cu -o bin/cuda.03.floats

cuda02: 
	/usr/local/cuda-7.5/bin/nvcc -m64 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50 -lineinfo -Xptxas=-v cuda.02.adjoint.cu -o bin/cuda.02.adjoint

cuda01: 
	/usr/local/cuda-7.5/bin/nvcc -m64 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50 -lineinfo -Xptxas=-v cuda.01.adjoint.cu -o bin/cuda.01.adjoint

omp: 
	g++-4.6 -std=c++0x -O3 -g -fopenmp -Werror omp.adjoint.cpp -o bin/omp.adjoint

serial: 
	g++-4.6 -std=c++0x -O3 -g -Werror serial.adjoint.cpp -o bin/serial.adjoint

tests:
	g++ -std=c++0x -O3 -g -Werror testbed.cpp -o testbed
