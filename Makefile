suport.o: src/suport.cu
		nvcc --compile src/suport.cu -o src/suport.o -I ../libgputk/ -I./

infoGPU.o: infoGPU.cu
		nvcc --compile infoGPU.cu -I ../libgputk/ -I./ 

infoGPU: infoGPU.o
		nvcc -o infoGPU -lm -lcuda -lrt infoGPU.o suport.o

