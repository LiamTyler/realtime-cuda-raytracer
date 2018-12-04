#pragma once

#include "common.h"
#include "shapes.h"
#include "tinyobjloader/tiny_obj_loader.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

class Vertex {
    public:
        Vertex(const glm::vec3& vert, const glm::vec3& norm, const glm::vec2& tex) :
            vertex(vert),
            normal(norm),
            uv(tex) {}

        bool operator==(const Vertex& other) const {
            return vertex == other.vertex && normal == other.normal && uv == other.uv;
        }

        glm::vec3 vertex;
        glm::vec3 normal;
        glm::vec2 uv;
};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.vertex) ^
                        (hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^
                (hash<glm::vec2>()(vertex.uv) << 1);
        }
    };
}        

CudaMesh LoadOBJ(const std::string& filename) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, fullPath.c_str(), std::string(PG_RESOURCE_DIR).c_str(), true);

    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (!ret) {
        std::cout << "Failed to load the input file: " << fullPath << std::endl;
        return nullptr;
    }

    auto model = std::make_shared<Model>();

    for (int currentMaterialID = -1; currentMaterialID < (int) materials.size(); ++currentMaterialID) {
        std::shared_ptr<Material> currentMaterial;
        if (currentMaterialID == -1) {
            currentMaterial = ResourceManager::GetMaterial("default");
        } else {
            tinyobj::material_t& mat = materials[currentMaterialID];
            glm::vec3 ambient(mat.ambient[0], mat.ambient[1], mat.ambient[2]);
            glm::vec3 diffuse(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
            glm::vec3 specular(mat.specular[0], mat.specular[1], mat.specular[2]);
            glm::vec3 emissive(mat.emission[0], mat.emission[1], mat.emission[2]);
            float shininess = mat.shininess;
            Texture* diffuseTex = nullptr;
            if (mat.diffuse_texname != "") {
                diffuseTex = new Texture(new Image(PG_RESOURCE_DIR + mat.diffuse_texname), true, true, true);
            }

            currentMaterial = std::make_shared<Material>(ambient, diffuse, specular, emissive, shininess, diffuseTex);
        }

        std::vector<glm::vec3> verts;
        std::vector<glm::vec3> normals;
        std::vector<glm::vec2> uvs;
        std::vector<unsigned int> indices;
        std::unordered_map<Vertex, uint32_t> uniqueVertices = {};
        for (const auto& shape : shapes) {
            // Loop over faces(polygon)
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
                if (shape.mesh.material_ids[f] == currentMaterialID) {
                    // Loop over vertices in the face. Each face should have 3 vertices from the LoadObj triangulation
                    for (size_t v = 0; v < 3; v++) {
                        tinyobj::index_t idx = shape.mesh.indices[3 * f + v];
                        tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                        tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                        tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
                        //verts.emplace_back(vx, vy, vz);

                        tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
                        tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
                        tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];
                        //normals.emplace_back(nx, ny, nz);

                        tinyobj::real_t tx = 0, ty = 0;
                        if (idx.texcoord_index != -1) {
                            tx = attrib.texcoords[2 * idx.texcoord_index + 0];
                            ty = attrib.texcoords[2 * idx.texcoord_index + 1];
                            //uvs.emplace_back(tx, ty);
                        }


                        Vertex vertex(glm::vec3(vx, vy, vz), glm::vec3(nx, ny, nz), glm::vec2(ty, ty));
                        if (uniqueVertices.count(vertex) == 0) {
                            uniqueVertices[vertex] = static_cast<uint32_t>(verts.size());
                            verts.emplace_back(vx, vy, vz);
                            normals.emplace_back(nx, ny, nz);
                            if (idx.texcoord_index != -1)
                                uvs.emplace_back(tx, ty);
                        }


                        indices.push_back(uniqueVertices[vertex]);                            
                    }
                }
            }
        }

        // create mesh and upload to GPU
        if (verts.size()) {
            // TODO: make this work for meshes that dont have UVS
            glm::vec2* texCoords = nullptr;
            if (uvs.size())
                texCoords = &uvs[0];
            auto currentMesh = std::make_shared<Mesh>(verts.size(), indices.size() / 3, &verts[0], &normals[0], texCoords, &indices[0]);
            currentMesh->UploadToGPU(true, false);

            model->meshes.push_back(currentMesh);
            model->materials.push_back(currentMaterial);
        }
    }

    return model;
}
