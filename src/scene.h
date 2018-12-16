#pragma once

#include "progression.h"
#include "shapes.h"
#include "material.h"
#include "resource_loader.h"
#include "skybox.h"
#include <algorithm>

typedef struct RTObject {
    RTObject() :
        position(make_float3(0, 0, 0)),
        type(0),
        index(0)
    {
    }

    RTObject(float3 pos, int t, int i) :
        position(pos),
        type(t),
        index(i)
    {
    }

    float3 position;
    int type;
    int index;
} RTObject;

typedef struct RTScene {
    float3* lights;
    Sphere* spheres;
    CudaMesh* meshes;
    RTMaterial* materials;
    RTObject* objects;
    int numDirectionalLights;
    int numPointLights;
    int numSpheres;
    int numMeshes;
    int numObjects;
    RTSkybox skybox;
} RTScene;

int getMatID(const std::string& name, std::vector<RTMaterial>& mats, std::vector<std::string>& matNames) {
    auto iter = std::find(matNames.begin(), matNames.end(), name);
    int matID = 0;
    if (iter != matNames.end()) {
        matID = iter - matNames.begin();
    } else {
        matID = matNames.size();
        matNames.push_back(name);
        auto mat = ResourceManager::GetMaterial(name);
        if (!mat) {
            std::cout << "No material: " << name << " found in scene file" << std::endl;
            exit(EXIT_FAILURE);
        }
        mats.emplace_back(mat->diffuse, mat->specular, mat->emissive, mat->shininess, mat->ior);
    }
    return matID;
}

RTScene createRTSceneFromPGScene(Scene& pgScene) {
    RTScene scene;
    std::vector<std::string> meshNames;
    std::vector<std::string> matNames;
    std::vector<Sphere> spheres;
    std::vector<CudaMesh> meshes;
    std::vector<RTMaterial> materials;
    std::vector<RTObject> objects;
    std::vector<float3> lights;

    const auto& gameObjects = pgScene.GetGameObjects();
    for (auto& g : gameObjects) {
        auto rtc = g->GetComponent<RTInfo>();
        RTObject obj;
        obj.type = rtc->type;
        if (obj.type == 0) {
            const auto& p = g->transform.position;
            float3 pos = make_float3(p.x, p.y, p.z);
            spheres.emplace_back(pos, rtc->radius, getMatID(rtc->matName, materials, matNames));
        } else {
            const auto& list =  loadRTModel(PG_ROOT_DIR "../../" + rtc->modelName);
            if (list.size() != 1) {
                std::cout << "Loading model (" <<  rtc->modelName << ") with more than one mesh, not allowed" << std::endl;
                exit(EXIT_FAILURE);
            }
            for (auto pair : list) {
                if (rtc->matName != "") {
                    pair.first.matID = getMatID(rtc->matName, materials, matNames);
                } else {
                    pair.first.matID = materials.size();
                    materials.push_back(pair.second);
                    matNames.push_back("placeHolderName!!!@#$$");
                }
                meshes.push_back(pair.first);
                obj.index = meshes.size() - 1;
            }
        }

        objects.push_back(obj);
    }

    auto bc = pgScene.GetBackgroundColor();
    std::cout << "bg color = " << bc << std::endl;
    scene.skybox.bgColor = make_float3(bc.x, bc.y, bc.z);
    auto si = pgScene.getRTSkybox();
    if (si.left != "") {
        scene.skybox = RTSkybox(si.left, si.right, si.top, si.bottom, si.front, si.back);
    }

    // parse lights
    int numLights = pgScene.GetNumPointLights() + pgScene.GetNumDirectionalLights();
    const auto& dirLights = pgScene.GetDirectionalLights();
    const auto& pointLights = pgScene.GetDirectionalLights();
    for (const auto& l : dirLights) {
        glm::vec3 dir(0, 0, -1);
        glm::mat4 rot(1);
        rot = glm::rotate(rot, l->transform.rotation.z, glm::vec3(0, 0, 1));
        rot = glm::rotate(rot, l->transform.rotation.y, glm::vec3(0, 1, 0));
        rot = glm::rotate(rot, l->transform.rotation.x, glm::vec3(1, 0, 0));
        dir = glm::vec3(rot * glm::vec4(dir, 0));
        glm::vec3 color = l->intensity * l->color;

        lights.push_back(make_float3(dir.x, dir.y, dir.z));
        lights.push_back(make_float3(color.x, color.y, color.z));
    }
    for (const auto& l : pointLights) {
        glm::vec3 pos = l->transform.position;
        glm::vec3 color = l->intensity * l->color;

        lights.push_back(make_float3(pos.x, pos.y, pos.z));
        lights.push_back(make_float3(color.x, color.y, color.z));
    }
    std::cout << "num lights: " << numLights << std::endl;


    check(cudaMalloc((void**) &scene.lights, lights.size() * sizeof(float3)));
    check(cudaMalloc((void**) &scene.spheres, spheres.size() * sizeof(Sphere)));
    check(cudaMalloc((void**) &scene.materials, materials.size() * sizeof(RTMaterial)));
    check(cudaMalloc((void**) &scene.meshes, meshes.size() * sizeof(CudaMesh)));
    check(cudaMalloc((void**) &scene.objects, gameObjects.size() * sizeof(RTObject)));
    scene.numDirectionalLights = pgScene.GetNumDirectionalLights();
    scene.numPointLights = pgScene.GetNumPointLights();
    scene.numSpheres = spheres.size();
    scene.numMeshes = meshes.size();
    scene.numObjects = objects.size();

    check(cudaMemcpy(scene.lights, &lights[0], sizeof(float3) * lights.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(scene.spheres, &spheres[0], sizeof(Sphere) * spheres.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(scene.meshes, &meshes[0], sizeof(CudaMesh) * meshes.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(scene.materials, &materials[0], sizeof(RTMaterial) * materials.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(scene.objects, &objects[0], sizeof(RTObject) * objects.size(), cudaMemcpyHostToDevice));

    return scene;
}
