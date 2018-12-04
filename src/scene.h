#pragma once

#include "shapes.h"
#include "material.h"

typedef struct RTScene {
    float3* lights;
    int numDirectionalLights;
    int numPointLights;
    Sphere* spheres;
    int numSpheres;
    CudaMesh mesh;
    RTMaterial* materials;
} RTScene;
