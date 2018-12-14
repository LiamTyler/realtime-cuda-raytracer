#include "raytracer.h"

#define QSIZE 5;
#define BLOCK_SIZE 8

typedef struct QItem {
    __device__ QItem() {}
    __device__ QItem(const Ray& r, const float3& m, int d) : ray(r), multiplier(m), depth(d) {}
    Ray ray;
    float3 multiplier;
    int depth;
} QItem;

typedef struct RayQ {
    __device__ RayQ() : start(0), end(0) {}
    __device__ void push(const QItem& item) {
        Q[end] = item;
        end = (end + 1) % QSIZE;
    }

    __device__ bool pop(QItem& item) {
        if (start == end)
            return false;
        item = Q[start];
        start = (start + 1) % QSIZE;
        return true;
    }

    int start;
    int end;
    QItem Q[5];
} RayQ;

__device__
int intersection(const RTScene& scene, const Ray& ray, float& t, int& type, int& meshNum, float& u, float& v, int* localStack) {
    float minT = 1e30f;
    int index = -1;
    type = 0;
    for (int i = 0; i < scene.numSpheres; ++i) {
        if (raySphereTest(ray, scene.spheres[i], t)) {
            if (t < minT) {
                index = i;
                minT = t;
            }
        }
    }

    /*
    float uu, vv;
    for (int m = 0; m < scene.numMeshes; ++m) {
        CudaMesh& mesh = scene.meshes[m];
        for (int i = 0; i < mesh.numTriangles; ++i) {
            if (rayTriangleTest(ray, mesh, mesh.triangles[i], t, uu, vv)) {
                if (t < minT) {
                    meshNum = m; index = i; type = 1; minT = t; u = uu; v = vv;
                }
            }
        }
    }
    */

    float uu, vv;
    float3 invRayDir = 1.0f / ray.dir;
    int stack[64];
    for (int m = 0; m < scene.numMeshes; ++m) {
        CudaMesh& mesh = scene.meshes[m];
        BVH* bvh = mesh.bvh;
        int idx = 0;
        stack[idx++] = 0;

        while (idx) {
            int i = stack[--idx];
            // BVH node = bvh[i];
            BVH node;
            float4 f1  = tex1Dfetch<float4>(mesh.bvhTex, 2 * i + 0);
            float4 f2  = tex1Dfetch<float4>(mesh.bvhTex, 2 * i + 1);
            node.min   = make_float3(f1.x, f1.y, f1.z);
            node.max   = make_float3(f2.x, f2.y, f2.z);
            node.left  = *(int*) &f1.w;
            node.right = *(int*) &f2.w;
            
            if (!RayAABBTest2(ray.pos, invRayDir, node.min, node.max, minT))
                continue;

            // if not a leaf node
            if (!node.isLeaf()) {
                if (node.left)
                    stack[idx++] = node.left;
                if (node.right)
                    stack[idx++] = node.right;
            } else { // if leaf
                Triangle leftTri = mesh.triangles[-node.left];
                if (rayTriangleTest2(ray, mesh, leftTri, t, uu, vv)) {
                    if (t < minT) {
                        meshNum = m; index = -node.left; type = 1; minT = t; u = uu; v = vv;
                    }
                }
                if (node.right < 0) {
                    Triangle rightTri = mesh.triangles[-node.right];
                    if (rayTriangleTest2(ray, mesh, rightTri, t, uu, vv)) {
                        if (t < minT) {
                            meshNum = m; index = -node.right; type = 1; minT = t; u = uu; v = vv;
                        }
                    }
                }
            }
        }
    }

    t = minT;
    return index;
}

__device__ float3 computeLighting(const RTScene& scene, const RTMaterial& mat, const float3& P, const float3& N, const float3& V, float t) {
    float3 color = make_float3(0, 0, 0);

    for (int i = 0; i < scene.numDirectionalLights; ++i) {
        float3 l = scene.lights[2 * i];
        float3 lightColor = scene.lights[2 * i + 1];

        color += lightColor * mat.kd * fmaxf(0.0f, dot(N, -l));
        color += lightColor * mat.ks * powf(fmaxf(0.0f, dot(V, reflect(l, N))), mat.power);
    }

    return color;
}

__device__ float3 traceRay(RayQ& Q, const QItem& item, const RTScene& scene, int* localStack) {
    if (item.depth >= 5)
        return make_float3(0, 0, 0);

    float t;
    const Ray& ray = item.ray;
    int type, meshNum;
    float u, v;
    int index = intersection(scene, ray, t, type, meshNum, u, v, localStack);
    if (index == -1)
        return make_float3(0, 0, 0);

    float3 color = make_float3(0, 0, 0);
    float3 p = ray.eval(t);
    float3 n;
    unsigned short matID;

    if (type == 0) { // sphere
        const Sphere& s = scene.spheres[index];
        n = normalize(p - s.pos);
        matID = s.matID;
    } else { // triangle
        n = scene.meshes[meshNum].getNormal(index, u, v);
        // if (dot(n, ray.dir) > 0)
        //     n = -n;
        matID = scene.meshes[meshNum].matID;
    }
    const RTMaterial& mat = scene.materials[matID];

    color += item.multiplier * computeLighting(scene, mat, p, n, -ray.dir, t);

    float3 reflectMult = item.multiplier * mat.ks;
    if (dot(reflectMult, reflectMult) < 0.1f)
        return color;

    float3 reflectDir = reflect(ray.dir, n);
    Ray reflectRay(p + 0.001f * reflectDir, reflectDir);
    QItem reflectItem(reflectRay, reflectMult, item.depth + 1);
    Q.push(reflectItem);

    return color;
}

__global__
void rayTraceKernel(cudaSurfaceObject_t surface, int SW, int SH,
        float3 P, float3 UL, float3 DX, float3 DY, RTScene scene) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    //
    // __shared__ int stacks[32*BLOCK_SIZE*BLOCK_SIZE];
    // int* localStack = &stacks[32 * (BLOCK_SIZE*threadIdx.y + threadIdx.x)];
    int* localStack = NULL;

    if (x >= SW && y >= SH)
        return;

    RayQ Q;

    float3 pos2 = UL + x * DX + y * DY;
    QItem item(Ray(P, normalize(pos2 - P)), make_float3(1, 1, 1), 0);
    Q.push(item);

    float3 color = make_float3(0, 0, 0);

    while (Q.pop(item)) {
        color += traceRay(Q, item, scene, localStack);
    }

    uchar4 pixel = toPixel(color);
    surf2Dwrite(pixel, surface, x * sizeof(uchar4), y);
}

void RayTracer::Init(Scene& pgScene) {
    copyShader_ = PG::Shader(PG_RESOURCE_DIR "shaders/copy.vert", PG_RESOURCE_DIR "shaders/copy.frag");

    int SW = PG::Window::getWindowSize().x;
    int SH = PG::Window::getWindowSize().y;

    float quadVerts[] = {
        -1, 1,
        -1, -1,
        1, -1,

        -1, 1,
        1, -1,
        1, 1
    };
    glGenVertexArrays(1, &quadVAO_);
    glBindVertexArray(quadVAO_);
    glGenBuffers(1, &quadVBO_);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(copyShader_["vertex"]);
    glVertexAttribPointer(copyShader_["vertex"], 2, GL_FLOAT, GL_FALSE, 0, 0);

    glGenTextures(1, &glTexture_);
    glBindTexture(GL_TEXTURE_2D, glTexture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, SW, SH, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    check(cudaGraphicsGLRegisterImage(&cudaTex_, glTexture_, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    scene = createRTSceneFromPGScene(pgScene);
}

void RayTracer::Free() {
    glDeleteBuffers(1, &quadVBO_);
    glDeleteVertexArrays(1, &quadVAO_);
    glDeleteTextures(1, &glTexture_);
}

void RayTracer::Render(Camera* camera) {
    int SW = PG::Window::getWindowSize().x;
    int SH = PG::Window::getWindowSize().y;

    // glBindTexture(GL_TEXTURE_2D, glTexture_);
    check(cudaGraphicsMapResources(1, &cudaTex_));

    cudaArray_t texture_ptr;
    check(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cudaTex_, 0, 0))

    struct cudaResourceDesc description;
    memset(&description, 0, sizeof(description));
    description.resType = cudaResourceTypeArray;
    description.res.array.array = texture_ptr;

    cudaSurfaceObject_t surf;
    check(cudaCreateSurfaceObject(&surf, &description));

    auto p = camera->transform.position;
    auto dir = camera->GetForwardDir();
    auto up = camera->GetUpDir();
    auto right = camera->GetRightDir();
    float fov = camera->GetFOV();

    float d = SH / (2.0f * tan(fov));
    glm::vec3 ul = p + d * dir + up * (SH / 2.0f) - (SW / 2.0f) * right;

    float3 P, UL, DX, DY;
    P = make_float3(p.x, p.y, p.z);
    DX = make_float3(right.x, right.y, right.z);
    DY = make_float3(-up.x, -up.y, -up.z);
    UL = make_float3(ul.x, ul.y, ul.z);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim;
    gridDim.x = SW / blockDim.x + ((SW % blockDim.x) != 0);
    gridDim.y = SH / blockDim.y + ((SH % blockDim.y) != 0);
    gridDim.z = 1; 

    // std::cout << "num spheres: " << scene.numSpheres << std::endl;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("kernel Error: %s\n", cudaGetErrorString(err));
    }
    rayTraceKernel<<<gridDim, blockDim>>>(surf, SW, SH, P, UL, DX, DY, scene);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("kernel Error: %s\n", cudaGetErrorString(err));
    }

    check(cudaDestroySurfaceObject(surf));
    check(cudaGraphicsUnmapResources(1, &cudaTex_));
    // cudaStreamSynchronize(0);
    check(cudaDeviceSynchronize());


    PG::graphics::SetClearColor(1, 1, 1, 1);
    PG::graphics::Clear();
    copyShader_.Enable();
    glBindVertexArray(quadVAO_);
    graphics::Bind2DTexture(glTexture_, copyShader_["tex"], 0);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

