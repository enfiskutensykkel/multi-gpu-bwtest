PROJECT := bwtest
OBJECTS	:= main.o bench.o buffer.o stream.o timer.o
DEPS	:= buffer.h stream.h timer.h bench.h
CFLAGS  := -Wall -Wextra -pedantic
NVCC    := /usr/local/cuda/bin/nvcc

ifeq ($(shell uname -s),Darwin)
CCDIR	:= /Library/Developer/CommandLineTools/usr/bin/
CFLAGS  += -Wno-gnu-designator -Wno-c99-extensions -Wno-language-extension-token
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
