Measure bandwidth simultanously
===============================================================================
As the CUDA samples program `bandwidthTest` is insufficient for measuring
the bandwidth to multiple CUDA devices, this program uses CUDA streams in
order to attempt to start multiple simultanous `cudaMemcpyAsync()` transfers.

To verify that transfers were started at the same time, use the NVIDIA
Visual Profiler (`nvvp`). Judging by our preliminary results, however, it
seems that the CUDA driver will only transfer simultaneously if the transfer
is slow enough.


Requirements and remarks
-------------------------------------------------------------------------------
  - One or more CUDA capabable devices (NVIDIA GPUs)
  - CUDA version 7.0 or higher
  - GCC or Clang with support for C++11
  - Make (for example GNU Make)

I've tested it on both Mac OS X 10.11.3 and 10.11.4, as well as 
Ubuntu 14.04.04. It compiles with GCC version 4.8.4 and LLVM 7.0.2 and 
LLVM 7.2.0.

I've also noticed that using NVIDIA GPUs with compute capability lower 
than 3.5 messes up the results because CUDA is not able to start things in
parallel for some reason.

Usage
-------------------------------------------------------------------------------
```
Usage: ./bwtest --do=<transfer specs>... [--streams=<mode>]

Description
    This program uses multiple CUDA streams in an attempt at optimizing data
    transfers between host and multiple CUDA devices using cudaMemcpyAsync().

Program options
  --do=<transfer specs>    transfer specification
  --streams=<mode>         stream modes for transfers
  --list                   list available CUDA devices and quit
  --help                   show this help text and quit

Stream modes
  per-transfer             one stream per transfer [default]
  per-device               transfers to the same device share streams
  only-one                 all transfers share the same single stream

Transfer specification format
       <device>[:<direction>][:<size>][:<memory options>...]

Transfer specification arguments
  <device>                 CUDA device to use for transfer
  <direction>              transfer directions
  <size>                   transfer size in bytes [default is 32 MiB]
  <memory options>         memory allocation options

Transfer directions
  HtoD                     host to device transfer (RAM to GPU)
  DtoH                     device to host transfer (GPU to RAM)
  both                     first HtoD then DtoH [default]
  reverse                  first DtoH then HtoD

Memory options format
       option1,option2,option3,...

Memory options
  mapped                   map host memory into CUDA address space
  wc                       allocate write-combined memory on the host

```


Example runs
-------------------------------------------------------------------------------
```
jonas@alpha:~/multi-gpu-bwtest$ ./bwtest --list

 ID   Device name       IO addr     Compute   Managed   Unified   Mappable    #
-------------------------------------------------------------------------------
  0   Tesla K40c        08:00.0        3.5        yes       yes        yes    2
  1   Quadro K2200      03:00.0        5.0        yes       yes        yes    1
  2   Tesla K40c        04:00.0        3.5        yes       yes        yes    2
  3   GeForce GTX 750   04:01.0        5.0        yes       yes        yes    1
  4   Quadro K2000      04:02.0        3.0        yes       yes        yes    1

jonas@alpha:~/multi-gpu-bwtest$ ./bwtest --do=all:1024:both
Allocating buffers......DONE
Executing transfers.....DONE
Synchronizing streams...DONE

=====================================================================================
 ID   Device name       Transfer size   Direction   Time elapsed   Bandwidth 
-------------------------------------------------------------------------------------
  0   Tesla K40c             1.00 KiB        HtoD          30 µs         34.15 MiB/s 
  1   Quadro K2200           1.00 KiB        HtoD          37 µs         27.75 MiB/s 
  2   Tesla K40c             1.00 KiB        HtoD          46 µs         22.44 MiB/s 
  3   GeForce GTX 750        1.00 KiB        HtoD          51 µs         20.18 MiB/s 
  4   Quadro K2000           1.00 KiB        HtoD          23 µs         43.84 MiB/s 
  0   Tesla K40c             1.00 KiB        DtoH          34 µs         30.05 MiB/s 
  1   Quadro K2200           1.00 KiB        DtoH          26 µs         39.41 MiB/s 
  2   Tesla K40c             1.00 KiB        DtoH          43 µs         23.92 MiB/s 
  3   GeForce GTX 750        1.00 KiB        DtoH          35 µs         29.52 MiB/s 
  4   Quadro K2000           1.00 KiB        DtoH          19 µs         54.61 MiB/s 
=====================================================================================

Aggregated total time      :          343 µs
Aggregated total bandwidth :        29.86 MiB/s
Estimated elapsed time     :          412 µs
Timed total bandwidth      :        24.88 MiB/s

jonas@alpha:~/multi-gpu-bwtest$ ./bwtest --do=0:DtoH:$((64 * 1024 * 102)):mapped --do=2:1024:HtoD
Allocating buffers......DONE
Allocating buffers......DONE
Executing transfers.....DONE
Synchronizing streams...DONE

=====================================================================================
 ID   Device name       Transfer size   Direction   Time elapsed   Bandwidth 
-------------------------------------------------------------------------------------
  0   Tesla K40c             6.38 MiB        DtoH         659 µs      10146.99 MiB/s 
  2   Tesla K40c             1.00 KiB        HtoD          25 µs         41.13 MiB/s 
=====================================================================================

Aggregated total time      :          684 µs
Aggregated total bandwidth :      9778.98 MiB/s
Estimated elapsed time     :          698 µs
Timed total bandwidth      :      9576.82 MiB/s

```
