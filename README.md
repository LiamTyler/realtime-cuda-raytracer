# Cuda Real Time Ray Tracer

## Compilation + Installation
```
git clone --recursive https://github.com/LiamTyler/realtime-cuda-raytracer.git
cd realtime-cuda-raytracer
mkdir build
cd build
cmake ..
make -j
```

Note: This works for my Ubuntu 18.04 desktop with g++ 7.3.0 and nvcc 10.0.130

## Tools
The obj -> rtModel converter is in the tools directory of the Progression submodule.
Should be built automatically

## Running
```
./raytracer
```
