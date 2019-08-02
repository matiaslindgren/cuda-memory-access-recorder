# CUDA memory access recorder and visualizer

todo: screenshots here
![](screen.gif)

## Quickstart

Two kernel examples from [Programming Parallel Computers](http://ppc.cs.aalto.fi/ch4/) are available in `examples`.

To generate access patterns of a simple matrix computation:
```sh
cd examples/v0
make && ./bin/main
```
Start a local web server for the animation app:
```sh
cd ../../web && python3 -m http.server
```
Go to http://0.0.0.0:8000 and submit the generated `examples/v0/access-patterns.txt` file.
