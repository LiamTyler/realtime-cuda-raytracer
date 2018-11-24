#include "progression.h"
#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace Progression;

#define check(ans) { _check((ans), __FILE__, __LINE__); }

void _check(cudaError_t code, char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

inline __host__ __device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator*(const float& a, const float3& b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}
inline __host__ __device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float3 normalize(const float3& a) {
    float l = 1.0f / sqrtf(dot(a, a));
    return l * a;
}

typedef struct Sphere {
    __host__ __device__ Sphere() {
        pos = make_float3(0, 0, 0);
        radius = 1;
    }
    __host__ __device__ Sphere(float3 p, float r) {
        pos = p;
        radius = r;
    }
    float3 pos;
    float radius;
} Sphere;

typedef struct Ray {
    __host__ __device__ Ray(float3 p, float3 d) : pos(p), dir(d) {}
    float3 pos;
    float3 dir;
} Ray;

__host__ __device__ bool raySphereTest(const Ray& ray, const Sphere& sphere, float& t) {
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

__host__ __device__
unsigned int intersection(const Ray& ray, float& t) {
    Sphere s;
    s.pos = make_float3(0, 0, 0);
    s.radius = 2;
    if (raySphereTest(ray, s, t))
        return 1;

    return 0;
}

__global__
void rayTraceKernel(cudaSurfaceObject_t surface, int SW, int SH, float3 P, float3 UL, float3 DX, float3 DY) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= SW && y >= SH)
        return;

    float3 pos2 = UL + x * DX + y * DY;
    // Ray ray(P, pos2 - P);
    Ray ray(P, normalize(pos2 - P));
    float t;

    uchar4 pixel = make_uchar4(0, 0, 0, 255);
    if (intersection(ray, t))
        pixel = make_uchar4(255, 255, 255, 255);

    surf2Dwrite(pixel, surface, x * sizeof(uchar4), y);
}

bool Intersect(glm::vec3 rayP, glm::vec3 rayD, glm::vec3 sP, float sR, float& t) {
    float t0 = -1, t1 = -1;
    glm::vec3 OC = rayP - sP;
    float b = 2*glm::dot(rayD, OC);
    float c = glm::dot(OC,OC) - sR * sR;
    float disc = b*b - 4*c;
    if (disc < 0) {
        return false;
    }
    t0 = (-b + std::sqrt(disc)) / (2.0);
    t1 = (-b - std::sqrt(disc)) / (2.0);
    if (t0 > t1) {
        std::swap(t0, t1);
    }
    if (t1 < 0) {
        return false;
    }
    t = t0;
    if (t0 < 0)
        t = t1;

    return true;
}

int main(int argc, char* argv[]) {
    auto conf = PG::config::Config(PG_ROOT_DIR "configs/default.toml");
    if (!conf) {
        std::cout << "could not parse config file" << std::endl;
        exit(0);
    }

    PG::EngineInitialize(conf);

    auto scene = Scene::Load("/home/liam/Documents/School/5351/realtime-cuda-raytracer/rayTrace.pgscn");
    auto camera = scene->GetCamera();
    // camera->AddComponent<UserCameraComponent>(new UserCameraComponent(camera));

    int SW = Window::getWindowSize().x;
    int SH = Window::getWindowSize().y;

    Shader copyShader = Shader(PG_RESOURCE_DIR "shaders/copy.vert", PG_RESOURCE_DIR "shaders/copy.frag");

    float quadVerts[] = {
        -1, 1,
        -1, -1,
        1, -1,

        -1, 1,
        1, -1,
        1, 1
    };
    GLuint quadVAO;
    GLuint quadVBO;
    glGenVertexArrays(1, &quadVAO);
    glBindVertexArray(quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(copyShader["vertex"]);
    glVertexAttribPointer(copyShader["vertex"], 2, GL_FLOAT, GL_FALSE, 0, 0);

    GLuint glTexture;
    glGenTextures(1, &glTexture);
    glBindTexture(GL_TEXTURE_2D, glTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, SW, SH, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    cudaGraphicsResource_t cudaTex;
    check(cudaGraphicsGLRegisterImage(&cudaTex, glTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

    Sphere h_spheres[1];
    h_spheres[0] = Sphere(make_float3(0, 0, 0), 2);

    // check(cudaMalloc((void**) &d_spheres, 1 * sizeof(Sphere)));
    // check(cudaMemcpyToSymbol(d_spheres, h_spheres, sizeof(Sphere) * 1));
    // check(cudaMemcpy(d_spheres, h_spheres, sizeof(Sphere) * 1, cudaMemcpyHostToDevice));

    Window::SetRelativeMouse(true);
    PG::Input::PollEvents();
    while (!PG::EngineShutdown) {
        PG::Window::StartFrame();
        PG::Input::PollEvents();

        if (PG::Input::GetKeyDown(PG::PG_K_ESC))
            PG::EngineShutdown = true;

        glBindTexture(GL_TEXTURE_2D, glTexture);
        cudaArray_t texture_ptr;
        cudaGraphicsMapResources(1, &cudaTex, 0); 
        cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cudaTex, 0, 0);

        struct cudaResourceDesc description;
        memset(&description, 0, sizeof(description));
        description.resType = cudaResourceTypeArray;
        description.res.array.array = texture_ptr;

        cudaSurfaceObject_t surf;
        cudaCreateSurfaceObject(&surf, &description); 

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
        rayTraceKernel<<<gridDim, blockDim>>>(surf, SW, SH, P, UL, DX, DY);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("kernel Error: %s\n", cudaGetErrorString(err));
        }

        cudaDestroySurfaceObject(surf);
        cudaGraphicsUnmapResources(1, &cudaTex);
        cudaStreamSynchronize(0);



        // scene->Update();

        graphics::SetClearColor(1, 1, 1, 1);
        graphics::Clear();
        copyShader.Enable();
        glBindVertexArray(quadVAO);
        graphics::Bind2DTexture(glTexture, copyShader["tex"], 0);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        PG::Window::EndFrame();
    }

    PG::EngineQuit();

    return 0;
}
