 
CC = /usr/local/cuda/bin/nvcc
PROGS = forward
OBJS =
INC =
COMPILE_OPT = -lcublas -O3 -lcudnn

all: ${PROGS}

%: %.cu ${OBJS} ${INC} Makefile
	${CC} ${COMPILE_OPT} -o $@ ${OBJS} $<


clean :
	rm ${PROGS}
