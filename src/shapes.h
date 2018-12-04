#ifndef SHAPES_H_
#define SHAPES_H_

#include "common.h"

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

typedef struct Triangle {
    __host__ __device__ Triangle() {}

    __host__ __device__ Triangle(int a, int b, int c) : v1(a), v2(b), v3(c) {}

    int v1, v2, v3;
} Triangle;

typedef struct CudaMesh {

    __host__ CudaMesh() {}
    __host__ CudaMesh(const std::vector<glm::vec3>& verts,
                      const std::vector<glm::vec3>& norms,
                      const std::vector<Triangle>& tris,
                      unsigned short m) {
        matID = m;
        numTriangles = tris.size();

        check(cudaMalloc((void**) &vertices, verts.size() * sizeof(float3)));
        check(cudaMemcpy(vertices, &verts[0].x, verts.size() * sizeof(float3), cudaMemcpyHostToDevice));

        check(cudaMalloc((void**) &normals, norms.size() * sizeof(float3)));
        check(cudaMemcpy(normals, &norms[0].x, norms.size() * sizeof(float3), cudaMemcpyHostToDevice));

        check(cudaMalloc((void**) &triangles, tris.size() * sizeof(Triangle)));
        check(cudaMemcpy(triangles, &tris[0], tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    }

    __device__ void getVertices(const Triangle& t, float3& v1, float3& v2, float3& v3) const {
        v1 = vertices[t.v1];
        v2 = vertices[t.v2];
        v3 = vertices[t.v3];
    }
    __device__ void getNormals(const Triangle& t, float3& n1, float3& n2, float3& n3) const {
        n1 = normals[t.v1];
        n2 = normals[t.v2];
        n3 = normals[t.v3];
    }

    __device__ float3 getNormal(int index, float u, float v) const {
        float3 n1, n2, n3;
        getNormals(triangles[index], n1, n2, n3);

        return u * n1 + v * n2 + (1.0f - u - v) * n3;
    }

    float3* vertices;
    float3* normals;
    Triangle* triangles;
    unsigned short matID;
    int numTriangles;
} CudaMesh;


typedef struct Ray {
    __host__ __device__ Ray() : pos(make_float3(0, 0, 0)), dir(make_float3(0, 0, 0)) {}
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

__device__ bool rayTriangleTest(const Ray& ray, const CudaMesh& mesh, const Triangle& triangle, float& t) {
    float3 v1, v2, v3;
    mesh.getVertices(triangle, v1, v2, v3);

    float3 e12 = v2 - v1;
    float3 e13 = v3 - v1;
    float3 N = cross(e12, e13);

    float d = dot(v1, N);
    t = -(dot(ray.pos, N) - d) / dot(ray.dir, N);
    if (t < 0)
        return false;

    float3 P = ray.eval(t);
    float area = length(N) * 0.5f;
    float3 vp, vpCross;

    vp      = P - v1;
    vpCross = cross(e12, vp);
    if (dot(vpCross, N) < 0)
        return false;

    vp      = P - v2;
    vpCross = cross(v3 - v2, vp);
    if (dot(vpCross, N) < 0)
        return false;

    // float u = 0.5f * length(vpCross) / area;

    vp      = P - v3;
    vpCross = cross(v1 - v3, vp);
    if (dot(vpCross, N) < 0)
        return false;

    // float v = 0.5f * length(vpCross) / area;

    return true;
}

#endif
