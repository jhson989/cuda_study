CC = /usr/local/cuda/bin/nvcc
PROGS = matmul

all: ${PROGS}

%: %.cu
	${CC} -o $@ $<


clean :
	rm ${PROGS}


