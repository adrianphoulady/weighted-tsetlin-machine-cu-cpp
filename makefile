experiment: fastrand.cuh weightm.cuh multiweightm.cuh utils.cuh experiment.cu
	nvcc -std=c++11 -O3 -o experiment experiment.cu

clean:
	rm *.o experiment
