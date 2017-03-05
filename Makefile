serial: 
	g++ -std=c++0x -O3 -g -Werror serial.adjoint.cpp -o bin/serial.adjoint


tests:
	g++ -std=c++0x -O3 -g -Werror testbed.cpp -o testbed
