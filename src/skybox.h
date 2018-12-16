#pragma once

#include "common.h"

typedef struct RTSkybox {
    RTSkybox() : bgColor(make_float3(0, 0, 0)), width(0), height(0), data(NULL) {}
    RTSkybox(
            const std::string& left,
            const std::string& right,
            const std::string& top,
            const std::string& bottom,
            const std::string& front,
            const std::string& back)
    {
        std::vector<std::string> faces = { right, left, top, bottom, front, back };
        std::vector<Image> images;
        images.resize(6);
        for (int i = 0; i < 6; i++) {
            if (!images[i].LoadImage("/home/liam/Documents/Progression/resources/" + faces[i])) {
                std::cout << "Failed to load skybox texture: " << faces[i] << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        width = images[0].Width();
        height = images[0].Height();
        int w = width;
        int h = height;
        int size = 6 * 4 * w * h;

        check(cudaMalloc((void**) &data, size));

        for (int i = 0; i < 6; i++) {
            unsigned char* imgData = images[i].GetData();
            check(cudaMemcpy(data + i * w * h, imgData, 4 * w * h, cudaMemcpyHostToDevice));
        }
    }

    __device__ float3 getColor(const float3& dir) const {
        if (data == NULL) {
            return bgColor;
        }
        float absX = fabsf(dir.x);
        float absY = fabsf(dir.y);
        float absZ = fabsf(dir.z);

        bool isXPositive = dir.x > 0;
        bool isYPositive = dir.y > 0;
        bool isZPositive = dir.z > 0;

        float maxAxis, uc, vc;
        int index;

        // POSITIVE X
        if (isXPositive && absX >= absY && absX >= absZ) {
            // u (0 to 1) goes from +z to -z
            // v (0 to 1) goes from -y to +y
            maxAxis = absX;
            uc = -dir.z;
            vc = dir.y;
            index = 0;
        }
        // NEGATIVE X
        else if (!isXPositive && absX >= absY && absX >= absZ) {
            // u (0 to 1) goes from -z to +z
            // v (0 to 1) goes from -y to +y
            maxAxis = absX;
            uc = dir.z;
            vc = dir.y;
            index = 1;
        }
        // POSITIVE Y
        else if (isYPositive && absY >= absX && absY >= absZ) {
            // u (0 to 1) goes from -x to +x
            // v (0 to 1) goes from +z to -z
            maxAxis = absY;
            uc = dir.x;
            vc = -dir.z;
            index = 2;
        }
        // NEGATIVE Y
        else if (!isYPositive && absY >= absX && absY >= absZ) {
            // u (0 to 1) goes from -x to +x
            // v (0 to 1) goes from -z to +z
            maxAxis = absY;
            uc = dir.x;
            vc = dir.z;
            index = 3;
        }
        // POSITIVE Z
        else if (isZPositive && absZ >= absX && absZ >= absY) {
            // u (0 to 1) goes from -x to +x
            // v (0 to 1) goes from -y to +y
            maxAxis = absZ;
            uc = dir.x;
            vc = dir.y;
            index = 4;
        }
        // NEGATIVE Z
        else {
        // if (!isZPositive && absZ >= absX && absZ >= absY) {
            // u (0 to 1) goes from +x to -x
            // v (0 to 1) goes from -y to +y
            maxAxis = absZ;
            uc = -dir.x;
            vc = dir.y;
            index = 5;
        }

        // Convert range from -1 to 1 to 0 to 1
        float u = 0.5f * (uc / maxAxis + 1.0f);
        float v = 0.5f * (-vc / maxAxis + 1.0f);
        int w = u * width;
        int h = v * height;
        uchar4 p = data[index * width * height + h * width + w];
        return make_float3(p.x / 255.0f, p.y / 255.0f, p.z / 255.0f);
    }

    float3 bgColor;
    int width;
    int height;
    uchar4* data;
} RTSkybox;
