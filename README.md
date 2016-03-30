Measure bandwidth simultanously
===============================
As the CUDA samples program `bandwidthTest` is insufficient for measuring
the bandwidth to multiple CUDA devices, this program uses CUDA streams in
order to attempt to start multiple simultanous `cudaMemcpyAsync()` transfers.

To verify that transfers were started at the same time, use the NVIDIA
Visual Profiler (`nvvp`). Judging by our preliminary results, however, it
seems that the CUDA driver will only transfer simultaneously if the transfer
is slow enough.


Usage
-------------------------------
```
Usage: ./bwtest --transfer=<transfer specs>... [--streams=<mode>] [--list] [--help]

Description
    This program uses multiple CUDA streams in an attempt at optimizing data
    transfers between host and multiple CUDA devices using cudaMemcpyAsync().

Program options
  --streams=<mode>     stream modes for transfers
  --list               list available CUDA devices and quit
  --help               show this help text and quit

Stream modes
  per-transfer         one stream per transfer [default]
  per-device           transfers to the same device share streams
  only-one             all transfers share the same single stream

Transfer specification format
    <device>[:<direction>][:<size>][:<memory options>...]

Transfer specification arguments
  <device>             CUDA device to use for transfer
  <direction>          transfer directions
  <size>               transfer size in bytes [default is 32 MiB]
  <memory options>     memory allocation options

Transfer directions
  HtoD                 host to device transfer (RAM to GPU)
  DtoH                 device to host transfer (GPU to RAM)
  both                 first HtoD then DtoH [default]
  reverse              first DtoH then HtoD

Memory options format
   option1,option2,option3,...

Memory options
  mapped               map host memory into CUDA address space
  managed              allocate managed memory on the device
  wc                   allocate write-combined memory on the host
```
