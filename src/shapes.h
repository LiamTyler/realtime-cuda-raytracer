#ifndef SHAPES_H_
#define SHAPES_H_

typedef struct Sphere {
    __host__ __device__ Sphere() :
        pos(make_float3(0, 0, 0)),
        radius(1),
        matID(0)
    {
    }

    __host__ __device__ Sphere(float3 p, float r, unsigned short m) : pos(p), radius(r), matID(m)
    {
    }

    float3 pos;
    float radius;
    unsigned short matID;
} Sphere;

typedef struct Ray {
    __host__ __device__ Ray(float3 p, float3 d) : pos(p), dir(d) {}
    float3 pos;
    float3 dir;

    __host__ __device__ float3 eval(float t) const {
        return pos + t * dir;
    }

} Ray;

__device__ bool raySphereTest(const Ray& ray, const Sphere& sphere, float& t) {
    float3 OC = sphere.pos - ray.pos;
    float b = dot(ray.dir, OC);
    float disc = b*b - dot(OC, OC) + sphere.radius * sphere.radius;
    if (disc < 0)
        return false;
    disc = sqrtf(disc);
    t = b - disc;
    if (t < 0)
        t = b + disc;
    return t >= 0;
}

#endif
