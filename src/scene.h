#pragma once

#include "shapes.h"
#include "material.h"

typedef struct RTScene {
    float3* lights;
    int numDirectionalLights;
    int numPointLights;
    Sphere* spheres;
    int numSpheres;
    RTMaterial* materials;
} RTScene;
