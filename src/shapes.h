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

typedef struct BVH {
    int numShapes;
    int left;
    int right;
    float3 min;
    float3 max;

    __device__ bool isLeaf() const { return numShapes != 0; }
} BVH;

typedef struct CudaMesh {

    __host__ CudaMesh() {}
    __host__ CudaMesh(const std::vector<glm::vec3>& verts,
                      const std::vector<glm::vec3>& norms,
                      const std::vector<Triangle>& tris,
                      const std::vector<BVH>& _bvh,
                      unsigned short m) {
        matID = m;
        numTriangles = tris.size();

        check(cudaMalloc((void**) &vertices, verts.size() * sizeof(float3)));
        check(cudaMemcpy(vertices, &verts[0].x, verts.size() * sizeof(float3), cudaMemcpyHostToDevice));

        check(cudaMalloc((void**) &normals, norms.size() * sizeof(float3)));
        check(cudaMemcpy(normals, &norms[0].x, norms.size() * sizeof(float3), cudaMemcpyHostToDevice));

        check(cudaMalloc((void**) &triangles, tris.size() * sizeof(Triangle)));
        check(cudaMemcpy(triangles, &tris[0], tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice));

        check(cudaMalloc((void**) &bvh, _bvh.size() * sizeof(BVH)));
        check(cudaMemcpy(bvh, &_bvh[0], _bvh.size() * sizeof(BVH), cudaMemcpyHostToDevice));
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
    BVH* bvh;
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

__device__ bool rayTriangleTest(
        const Ray& ray,
        const CudaMesh& mesh,
        const Triangle& triangle,
        float& t,
        float& u,
        float& v)
{
    float3 v1, v2, v3;
    mesh.getVertices(triangle, v1, v2, v3);

    float3 e12 = v2 - v1;
    float3 e13 = v3 - v1;
    float3 N = cross(e12, e13);

    float NdotRay = dot(ray.dir, N);
    if (NdotRay < 0.0001)
        return false;

    float d = dot(v1, N);
    t = -(dot(ray.pos, N) - d) / NdotRay;
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

    u = 0.5f * length(vpCross) / area;

    vp      = P - v3;
    vpCross = cross(v1 - v3, vp);
    if (dot(vpCross, N) < 0)
        return false;

    v = 0.5f * length(vpCross) / area;

    return true;
}

__device__ bool RayAABBTest(const float3& p, const float3& invDir, const float3& min, const float3& max) {
    float tmin = (min.x - p.x) * invDir.x;
    float tmax = (max.x - p.x) * invDir.x;
    float tmp;
    if (tmin > tmax) {
        tmp = tmax;
        tmax = tmin;
        tmin = tmp;
    }
    float tymin = (min.y - p.y) * invDir.y;
    float tymax = (max.y - p.y) * invDir.y;
    if (tymin > tymax) {
        tmp = tymax;
        tymax = tymin;
        tymin = tmp;
    }
    if ((tmin > tymax) || (tymin > tmax))
        return false;

    tmin = fmaxf(tymin, tmin);
    tmax = fminf(tymax, tmax);

    float tzmin = (min.z - p.z) * invDir.z;
    float tzmax = (max.z - p.z) * invDir.z;
    if (tzmin > tzmax) {
        tmp = tzmax;
        tzmax = tzmin;
        tzmin = tmp;
    }
    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    return true;

}

__device__ bool RayAABBTest2(const float3& p, const float3& invDir, const float3& min, const float3& max, const float& t) {
  float tx1 = (min.x - p.x)*invDir.x;
  float tx2 = (max.x - p.x)*invDir.x;

  float tmin = fminf(tx1, tx2);
  float tmax = fmaxf(tx1, tx2);

  float ty1 = (min.y - p.y)*invDir.y;
  float ty2 = (max.y - p.y)*invDir.y;

  tmin = fmaxf(tmin, fminf(ty1, ty2));
  tmax = fminf(tmax, fmaxf(ty1, ty2));

  float tz1 = (min.z - p.z)*invDir.z;
  float tz2 = (max.z - p.z)*invDir.z;

  tmin = fmaxf(tmin, fminf(tz1, tz2));
  tmax = fminf(tmax, fmaxf(tz1, tz2));

  return tmax >= fmaxf(0.0f, tmin) && tmin < t;
}

#endif
