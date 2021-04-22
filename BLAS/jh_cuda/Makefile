CC = /usr/local/cuda/bin/nvcc
PROGS = axpy

all: ${PROGS}

%: %.cu
	${CC} -o $@ $<


clean :
	rm ${PROGS}


