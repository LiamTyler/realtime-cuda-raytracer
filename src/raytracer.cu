#include "raytracer.h"

#define QSIZE 5;

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
int intersection(const RTScene& scene, const Ray& ray, float& t, int& type) {
    float minT;
    int index = -1;
    type = 0;
    for (int i = 0; i < scene.numSpheres; ++i) {
        if (raySphereTest(ray, scene.spheres[i], t)) {
            if (index == -1 || t < minT) {
                index = i;
                minT = t;
            }
        }
    }

    for (int i = 0; i < scene.mesh.numTriangles; ++i) {
        if (rayTriangleTest(ray, scene.mesh, scene.mesh.triangles[i], t)) {
            if (index == -1 || t < minT) {
                index = i;
                type = 1;
                minT = t;
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

__device__ float3 traceRay(RayQ& Q, const QItem& item, const RTScene& scene) {
    if (item.depth >= 5)
        return make_float3(0, 0, 0);

    float t;
    const Ray& ray = item.ray;
    int type;
    int index = intersection(scene, ray, t, type);
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
        n = scene.mesh.getNormal(index, 0, 0);
        matID = scene.mesh.matID;
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

    if (x >= SW && y >= SH)
        return;

    RayQ Q;

    float3 pos2 = UL + x * DX + y * DY;
    QItem item(Ray(P, normalize(pos2 - P)), make_float3(1, 1, 1), 0);
    Q.push(item);

    float3 color = make_float3(0, 0, 0);

    while (Q.pop(item)) {
        color += traceRay(Q, item, scene);
    }

    uchar4 pixel = toPixel(color);
    surf2Dwrite(pixel, surface, x * sizeof(uchar4), y);
}

void RayTracer::Init(int maxLights, int maxSpheres, int maxMaterials) {
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
    std::cout << "before register" << std::endl;

    check(cudaGraphicsGLRegisterImage(&cudaTex_, glTexture_, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    std::cout << "after register" << std::endl;

    check(cudaMalloc((void**) &scene.lights, 2 * maxLights * sizeof(float3)));
    check(cudaMalloc((void**) &scene.spheres, maxSpheres * sizeof(Sphere)));
    check(cudaMalloc((void**) &scene.materials, maxMaterials * sizeof(RTMaterial)));
    scene.numDirectionalLights = 0;
    scene.numPointLights = 0;
    scene.numSpheres = 0;


    std::cout << "after mallocs" << std::endl;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
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

    dim3 blockDim(16, 16, 1);
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

