#ifndef RAYTRACER_H_
#define RAYTRACER_H_

#include "common.h"
#include "shapes.h"
#include "scene.h"

class RayTracer {
    public:
        RayTracer() = default;

        void Init(Scene& pgScene);
        void Free();
        void Render(Camera* camera);

        RTScene scene;

    private:
        Shader copyShader_;
        GLuint quadVAO_, quadVBO_, glTexture_;
        cudaGraphicsResource_t cudaTex_;
};

#endif
