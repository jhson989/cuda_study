CC = /usr/local/cuda/bin/nvcc
PROGS = sgemm
OBJS =
INC =

all: ${PROGS}

%: %.cu ${OBJS} ${INC} Makefile
	${CC} -O3 -o $@ ${OBJS} $<


clean :
	rm ${PROGS}


