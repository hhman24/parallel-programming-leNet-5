#ifndef SUPPORT_H_INCLUDED
#define SUPPORT_H_INCLUDED
#pragma once

#include <stdio.h>
#include <stdint.h>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);                                                                 
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

void readPnm(char * fileName, int &numChannels, int &width, int &height, uint8_t * &pixels);
void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels);
void writePnm(uchar3 * pixels, int width, int height, char * fileName);
void writePnm(uint8_t * pixels, int numChannels, int width, int height, char * fileName);
void writePnm(uint32_t * pixels, int numChannels, int width, int height, 
		char * fileName);
char * concatStr(const char * s1, const char * s2);
void printDeviceInfo();

#endif