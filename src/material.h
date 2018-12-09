#pragma once

#include "common.h"

typedef struct RTMaterial {
    __host__ __device__ RTMaterial() :
        kd(make_float3(0, 0, 0)),
        ks(make_float3(0, 0, 0)),
        power(0),
        transmissive(make_float3(0, 0, 0)),
        ior(0)
    {
    }

    __host__ RTMaterial(const glm::vec3& d, const glm::vec3& s, const glm::vec3& t, float p, float i) {
        kd = make_float3(d.x, d.y, d.z);
        ks = make_float3(s.x, s.y, s.z);
        transmissive = make_float3(t.x, t.y, t.z);
        power = p;
        ior = i;
    }

    __host__ __device__ RTMaterial(const float3& d, const float3& s, float p, const float3& t, float i) :
        kd(d), ks(s), power(p), transmissive(t), ior(i)
    {
    }

    float3 kd;
    float pad;
    float3 ks;
    float power;
    float3 transmissive;
    float ior;
} RTMaterial;
