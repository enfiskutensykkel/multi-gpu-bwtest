PROJECT := bwtest
OBJECTS	:= main.o bench.o buffer.o stream.o timer.o device.o
DEPS	:= buffer.h stream.h timer.h bench.h device.h
CFLAGS  := -Wall -Wextra 
NVCC    := /usr/local/cuda/bin/nvcc

ifeq ($(shell uname -s),Darwin)
CCDIR	:= /Library/Developer/CommandLineTools/usr/bin/
CFLAGS  += -Wno-gnu-designator 
else
CCDIR   := /usr/bin/g++
endif

INCLUDE	:= /usr/local/cuda/include 

.PHONY: all clean $(PROJECT)

all: $(PROJECT)

clean:
	-$(RM) $(PROJECT) $(OBJECTS)

$(PROJECT): $(OBJECTS)
	$(NVCC) -ccbin $(CCDIR) -o $@ $^ 

%.o: %.cu $(DEPS)
	$(NVCC) -std=c++11 -x cu -ccbin $(CCDIR) -Xcompiler "$(CFLAGS)" $(addprefix -I,$(INCLUDE)) -o $@ $< -c 
