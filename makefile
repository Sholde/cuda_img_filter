CC=nvcc
CFLAGS=-Xcompiler -Wall
OFLAGS=-O3 --m64 -use_fast_math
LFLAGS=-lfreeimage -lm

.PHONY: all clean

TARGET=cuda_img_filter

all: $(TARGET)

$(TARGET): main.cu kernel.cu launcher.cu
	nvcc $(CFLAGS) $(OFLAGS) $(LFLAGS) $^ -o $@

main.cu: kernel.cu kernel.h launcher.cu launcher.h

kernel.cu: kernel.h launcher.cu launcher.h

launcher.cu: launcher.h

clean:
	rm -Rf *~ *.o $(TARGET)
