CC = /usr/local/cuda/bin/nvcc
PROGS = reduction
OBJS = 
INC = 

all: ${PROGS}

%: %.cu ${OBJS} ${INC} Makefile
	${CC} -O3 -o $@ ${OBJS} $<


clean :
	rm ${PROGS}


