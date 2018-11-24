#include "progression.h"
#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace Progression;

#define check(ans) { _check((ans), __FILE__, __LINE__); }

void _check(cudaError_t code, char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %S %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

__global__
void rayTraceKernel(cudaSurfaceObject_t surface, int width, int height) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        uchar4 pixel = make_uchar4(255, 255, 0, 255);
        surf2Dwrite(pixel, surface, x * sizeof(uchar4), y);
    }
}

int main(int argc, char* argv[]) {
    auto conf = PG::config::Config(PG_ROOT_DIR "configs/default.toml");
    if (!conf) {
        std::cout << "could not parse config file" << std::endl;
        exit(0);
    }

    PG::EngineInitialize(conf);

    glm::ivec2 WS(Window::getWindowSize().x, Window::getWindowSize().y);

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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, Window::getWindowSize().x, Window::getWindowSize().y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    cudaGraphicsResource_t cudaTex;
    check(cudaGraphicsGLRegisterImage(&cudaTex, glTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));


    // auto scene = Scene::Load(PG_ROOT_DIR "resources/scenes/bouncing_ball.pgscn");

    // auto camera = scene->GetCamera();
    // camera->AddComponent<UserCameraComponent>(new UserCameraComponent(camera));

    Window::SetRelativeMouse(true);
    PG::Input::PollEvents();

    // Game loop
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

        dim3 blockDim(16, 16, 1);
        dim3 gridDim;
        gridDim.x = WS.x / blockDim.x + ((WS.x % blockDim.x) != 0);
        gridDim.y = WS.y / blockDim.y + ((WS.y % blockDim.y) != 0);
        gridDim.z = 1; 
        rayTraceKernel<<<gridDim, blockDim>>>(surf, WS.x, WS.y);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error: %s\n", cudaGetErrorString(err));
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
