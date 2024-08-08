# Variable to set the compiler
CC = g++

# Flags for compiling C++ files
CFLAGS = -Wall -O3 -g -std=c++0x -larmadillo

all: NN-learn

# Build rule for the executable file
NN-learn: NN-learn.cpp neural_network.h circuit_simulation.h
	$(CC) $(CFLAGS) NN-learn.cpp -o NN-learn
