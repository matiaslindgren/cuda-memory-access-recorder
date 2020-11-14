# Examples

## Naive matrix multiplication

2304 thread blocks on 68 SMs.

![](gifs/screen-matmul.gif)

## Minimum shortcuts in a graph (4 versions)

Problem description [here](http://ppc.cs.aalto.fi/ch2/).

Baseline approach with a memory access pattern that uses many short cachelines, which leads to poor memory transaction coalescing ([source](http://ppc.cs.aalto.fi/ch4/v0/)).

4 thread blocks on 4 SMs.

![](gifs/screen-v0.gif)

Slightly adjusted access pattern where thread warps are accessing consecutive memory addresses, leading to fewer, wider memory transactions ([source](http://ppc.cs.aalto.fi/ch4/v1/)).

4 thread blocks on 4 SMs.

![](gifs/screen-v1.gif)

Reduced amount of memory accesses by reusing data in registers ([source](http://ppc.cs.aalto.fi/ch4/v2/)).
The input data has been copied and transposed to enable a linear memory access pattern for both row- and column-wise accesses.

1 thread block on 1 SM.

![](gifs/screen-v2.gif)

Buffering memory accesses through shared memory ([source](http://ppc.cs.aalto.fi/ch4/v3/)).

1 thread block on 1 SM.

![](gifs/screen-v3.gif)
