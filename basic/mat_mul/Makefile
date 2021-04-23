CC = /usr/local/cuda/bin/nvcc
PROGS = matmul
OBJS = kernel.o

all: ${PROGS}


%.o: %.cu
	${CC} -c -o $@ $<


%: %.cu
	${CC} -o $@ ${OBJS} $<


clean :
	rm ${PROGS} ${OBJS}


