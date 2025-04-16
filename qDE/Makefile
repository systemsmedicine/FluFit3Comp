HOST_COMPILER = gcc
NVCCFLAGS = -O2
LIBRARIES = -lm -lcurand
#NVCC = nvcc -ccbin $(HOST_COMPILER) $(NVCCFLAGS) 
NVCC = nvcc $(NVCCFLAGS)

OBJS = mainDEest.o

qDE : $(OBJS)
	$(NVCC) $^ -o $@ $(LIBRARIES)

%.o : %.cu
	$(NVCC) -c $< -o $@

clean:
	rm -f *.o *~

clean-all:
	rm -f *.o *~ *.dat
