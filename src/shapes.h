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

    __device__ void getVerts(float3& v1, float3& v2, float3& v3) const {
        v1 = make_float3(f1.x, f1.y, f1.z);
        v2 = make_float3(f1.w, f2.x, f2.y);
        v3 = make_float3(f2.z, f2.w, f3.x);
    }

    __device__ void getNormals(float3& n1, float3& n2, float3& n3) const {
        n1 = make_float3(f3.y, f3.z, f3.w);
        n2 = make_float3(f4.x, f4.y, f4.z);
        n3 = make_float3(f4.w, f5.x, f5.y);
    }

    float4 f1, f2, f3, f4, f5;
} Triangle;

typedef struct BVH {
    //int numShapes;
    float3 min;
    int left;
    float3 max;
    int right;

    __device__ bool isLeaf() const { return left < 0; }
} BVH;

typedef struct CudaMesh {

    __host__ CudaMesh() {}
    __host__ CudaMesh(
            const std::vector<Triangle>& tris,
            const std::vector<BVH>& _bvh,
            unsigned short m) {
        matID = m;
        numTriangles = tris.size();

        check(cudaMalloc((void**) &triangles, tris.size() * sizeof(Triangle)));
        check(cudaMemcpy(triangles, &tris[0], tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice));

        check(cudaMalloc((void**) &bvh, _bvh.size() * sizeof(BVH)));
        check(cudaMemcpy(bvh, &_bvh[0], _bvh.size() * sizeof(BVH), cudaMemcpyHostToDevice));

        // Specify texture
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
        resDesc.res.linear.desc.x = 32;
        resDesc.res.linear.desc.y = 32;
        resDesc.res.linear.desc.z = 32;
        resDesc.res.linear.desc.w = 32;
        resDesc.res.linear.devPtr = bvh;
        resDesc.res.linear.sizeInBytes = _bvh.size() * sizeof(BVH);

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;

        // Create texture objects
        check(cudaCreateTextureObject(&bvhTex, &resDesc, &texDesc, NULL));
    }

    __device__ float3 getNormal(int index, float u, float v) const {
        float3 n1, n2, n3;
        triangles[index].getNormals(n1, n2, n3);

        return (1.0f - u - v) * n1 + u * n2 + v * n3;
        // return u * n1 + v * n2 + (1.0f - u - v) * n3;
    }

    // float3* vertices;
    // float3* normals;
    Triangle* triangles;
    BVH* bvh;
    unsigned short matID;
    int numTriangles;
    cudaTextureObject_t bvhTex, triTex;
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
    triangle.getVerts(v1, v2, v3);

    float3 e12 = v2 - v1;
    float3 e13 = v3 - v1;
    float3 N = cross(e12, e13);

    float NdotRay = dot(ray.dir, N);
    if (NdotRay < EPSILON)
        return false;

    float d = dot(v1, N);
    t = -(dot(ray.pos, N) - d) / NdotRay;
    if (t < 0.0f)
        return false;

    float3 P = ray.eval(t);
    float area = length(N) * 0.5f;
    float3 vp, vpCross;

    vp      = P - v1;
    vpCross = cross(e12, vp);
    if (dot(vpCross, N) < 0.0f)
        return false;

    vp      = P - v2;
    vpCross = cross(v3 - v2, vp);
    if (dot(vpCross, N) < 0.0f)
        return false;

    u = 0.5f * length(vpCross) / area;

    vp      = P - v3;
    vpCross = cross(v1 - v3, vp);
    if (dot(vpCross, N) < 0.0f)
        return false;

    v = 0.5f * length(vpCross) / area;

    return true;
}

__device__ bool rayTriangleTest2(
        const Ray& ray,
        const CudaMesh& mesh,
        const Triangle& triangle,
        float& t,
        float& u,
        float& v,
        const float& minT)
{
    float3 v0, v1, v2;
    triangle.getVerts(v0, v1, v2);

    float3 v0v1 = v1 - v0;
    float3 v0v2 = v2 - v0;
    float3 pvec = cross(ray.dir, v0v2);
    float det = dot(v0v1, pvec);
    // ray and triangle are parallel if det is close to 0
    if (-EPSILON < det && det < EPSILON) return false;
    float invDet = 1.0f / det;

    float3 tvec = ray.pos - v0;
    u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return false;

    float3 qvec = cross(tvec, v0v1);
    v = dot(ray.dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return false;

    t = dot(v0v2, qvec) * invDet;
    return t < minT && t > EPSILON;
}

__device__ bool RayAABBTest(const float3& p, const float3& invDir, const float3& min, const float3& max, const float& t) {
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

    // return true;
    return tmax >= fmaxf(0.0f, tmin) && tmin < t;
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
