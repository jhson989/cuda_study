CC = /usr/local/cuda/bin/nvcc
PROGS = properties

all: ${PROGS}

%: %.cu
	${CC} -o $@ $<


clean :
	rm ${PROGS}


