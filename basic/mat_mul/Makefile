CC = /usr/local/cuda/bin/nvcc
PROGS = matmul
OBJS = kernel.cu
INC = kernel.hpp

all: ${PROGS}

%: %.cu ${OBJS} ${INC} Makefile
	${CC} -O3 -o $@ ${OBJS} $<


clean :
	rm ${PROGS}


