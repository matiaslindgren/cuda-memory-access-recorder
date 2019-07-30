# CUDA memory access recorder and visualizer

![](screen.gif)

## Quickstart

Generate access patterns of a simple matrix computation into `/tmp/access-patterns.txt`:
```sh
make && ./bin/main
```
Start a local web server for the animation app:
```sh
cd web && python3 -m http.server
```
Go to [http://0.0.0.0:8000]() and submit the generated `access-patterns.txt` file.
