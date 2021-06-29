
#include <cstdlib>
#include <cudnn.h>

#include "conv.cuh"

int main(int argc, char** argv) {


    Conv2d layer(1, 3, 3, 128, 128, 3, 3);

    return 0;
}