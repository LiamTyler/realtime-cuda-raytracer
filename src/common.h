#ifndef COMMON_H_
#define COMMON_H_

#include "progression.h"
#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace Progression;

#define check(ans) { _check((ans), __FILE__, __LINE__); }

void _check(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

inline __host__ __device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(float3& a, const float3& b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
}
inline __host__ __device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(float3& a, const float3& b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
inline __host__ __device__ float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}

inline __host__ __device__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ float3 operator*(const float& a, const float3& b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}
inline __host__ __device__ float3 operator*(const float3& b, const float& a) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}
inline __host__ __device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float3 normalize(const float3& a) {
    float l = 1.0f / sqrtf(dot(a, a));
    return l * a;
}
inline __host__ __device__ float3 reflect(const float3& D, const float3& N) {
    // return 2.0f * dot(D, N) * N - D;
    return D - 2.0f * dot(D, N) * N;
}
inline __host__ __device__ float3 fminf(const float3& a, const float3& b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
inline __host__ __device__ float3 fmaxf(const float3& a, const float3& b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

inline __host__ __device__ uchar4 toPixel(const float3& a) {
    float3 v = fmaxf(make_float3(0, 0, 0), fminf(a, make_float3(1, 1, 1)));
    return make_uchar4((unsigned char) 255 * v.x, (unsigned char) 255 * v.y, (unsigned char) 255 * v.z, 255);
}

#endif
