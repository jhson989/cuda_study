 
CC = /usr/local/cuda/bin/nvcc
PROGS = main
OBJS = conv.cu
INC = conv.cuh
COMPILE_OPT = -O3 -lcudnn

all: ${PROGS}

%: %.cu ${OBJS} ${INC} Makefile
	${CC} ${COMPILE_OPT} -o $@ ${OBJS} $<


clean :
	rm ${PROGS}
