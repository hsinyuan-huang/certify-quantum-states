# Variable to set the compiler
CC = g++

# Flags for compiling C++ files
CFLAGS = -Wall -O3 -g -std=c++0x -larmadillo

all: XEBvsShadowFid-Haar XEBvsShadowFid-Phase XEBvsShadowFid-Haar-general-err XEBvsShadowFid-Phase-general-err

# Build rule for the executable file
XEBvsShadowFid-Haar: XEBvsShadowFid-Haar.cpp circuit_simulation.h
	$(CC) $(CFLAGS) XEBvsShadowFid-Haar.cpp -o XEBvsShadowFid-Haar

XEBvsShadowFid-Phase: XEBvsShadowFid-Phase.cpp circuit_simulation.h
	$(CC) $(CFLAGS) XEBvsShadowFid-Phase.cpp -o XEBvsShadowFid-Phase

XEBvsShadowFid-Haar-general-err: XEBvsShadowFid-Haar-general-err.cpp circuit_simulation.h
	$(CC) $(CFLAGS) XEBvsShadowFid-Haar-general-err.cpp -o XEBvsShadowFid-Haar-general-err

XEBvsShadowFid-Phase-general-err: XEBvsShadowFid-Phase-general-err.cpp circuit_simulation.h
	$(CC) $(CFLAGS) XEBvsShadowFid-Phase-general-err.cpp -o XEBvsShadowFid-Phase-general-err
