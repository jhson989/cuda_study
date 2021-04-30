CC = /usr/local/cuda/bin/nvcc
PROGS = reduction_basic
OBJS = 
INC = 

all: ${PROGS}

%: %.cu ${OBJS} ${INC} Makefile
	${CC} -O3 -o $@ ${OBJS} $<


clean :
	rm ${PROGS}


