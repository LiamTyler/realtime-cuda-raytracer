#include "raytracer.h"

__device__
int intersection(const RTScene& scene, const Ray& ray, float& t) {
    float minT;
    int index = -1;
    for (int i = 0; i < scene.numSpheres; ++i) {
        if (raySphereTest(ray, scene.spheres[i], t)) {
            if (index == -1 || t < minT) {
                index = i;
                minT = t;
            }
        }
    }

    t = minT;
    return index;
}

__device__ float3 lightSphere(const RTScene& scene, const Sphere& sphere, const Ray& ray, float t) {
    float3 p = ray.eval(t);
    float3 n = normalize(p - sphere.pos);
    const RTMaterial& mat = scene.materials[sphere.matID];
    float3 color = make_float3(0, 0, 0);
    float3 v = -ray.dir;

    for (int i = 0; i < scene.numDirectionalLights; ++i) {
        float3 l = scene.lights[2 * i];
        float3 lightColor = scene.lights[2 * i + 1];

        color += lightColor * mat.kd * fmaxf(0.0f, dot(n, -l));
        color += lightColor * mat.ks * powf(fmaxf(0.0f, dot(v, reflect(l, n))), mat.power);
    }

    return color;
}

__device__ float3 traceRay(const Ray& ray, const RTScene& scene, int depth) {
    if (depth >= 5)
        return make_float3(0, 0, 0);

    float t;
    int index = intersection(scene, ray, t);
    if (index == -1)
        return make_float3(0, 0, 0);

    
    float3 color = make_float3(0, 0, 0);
    const Sphere& s = scene.spheres[index];

    color += lightSphere(scene, s, ray, t);

    float3 p = ray.eval(t);
    float3 n = normalize(p - s.pos);
    float3 reflectDir = reflect(ray.dir, n);
    Ray reflectRay(p + 0.001f * reflectDir, reflectDir);
    color += scene.materials[s.matID].ks * traceRay(reflectRay, scene, depth + 1);

    return color;
}

__global__
void rayTraceKernel(cudaSurfaceObject_t surface, int SW, int SH,
        float3 P, float3 UL, float3 DX, float3 DY, RTScene scene) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= SW && y >= SH)
        return;

    float3 pos2 = UL + x * DX + y * DY;
    Ray ray(P, normalize(pos2 - P));

    float3 color = traceRay(ray, scene, 0);
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

    cudaDestroySurfaceObject(surf);
    cudaGraphicsUnmapResources(1, &cudaTex_);
    // cudaStreamSynchronize(0);
    cudaDeviceSynchronize();


    PG::graphics::SetClearColor(1, 1, 1, 1);
    PG::graphics::Clear();
    copyShader_.Enable();
    glBindVertexArray(quadVAO_);
    graphics::Bind2DTexture(glTexture_, copyShader_["tex"], 0);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

