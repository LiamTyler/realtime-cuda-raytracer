#pragma once

#include "common.h"
#include "shapes.h"

std::ostream& operator<<(std::ostream& out, const float3& v) {
    return out << v.x << " " << v.y << " " << v.z;
}

std::vector<std::pair<CudaMesh, RTMaterial>> loadRTModel(const std::string& filename) {
    std::vector<std::pair<CudaMesh, RTMaterial>> list;
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cout << "Failed to load the input file: " << filename << std::endl;
        return {};
    }

    int numMeshes;
    in.read((char*)&numMeshes, sizeof(int));
    list.resize(numMeshes);
    std::cout << "num meshes: " << numMeshes << std::endl;

    // parse all of the materials
    for (int i = 0; i < numMeshes; ++i) {
        auto& mat = list[i].second;
        in.read((char*)&mat.kd, sizeof(glm::vec3));
        in.read((char*)&mat.ks, sizeof(glm::vec3));
        in.read((char*)&mat.transmissive, sizeof(glm::vec3));
        in.read((char*)&mat.power, sizeof(float));
        in.read((char*)&mat.ior, sizeof(float));

        std::cout << "KD = " << mat.kd << std::endl;
        std::cout << "KS = " << mat.ks << std::endl;
        std::cout << "KT = " << mat.transmissive << std::endl;
        std::cout << "power = " << mat.power << std::endl;
        std::cout << "ior = " << mat.ior << std::endl;
    }

    std::vector<glm::vec3> verts;
    std::vector<glm::vec3> normals;
    std::vector<Triangle> tris;
    std::vector<BVH> bvh;

    // parse all of the meshes
    for (int i = 0; i < numMeshes; ++i) {
        auto& mesh = list[i].first;

        int numVerts, numTris, bvh_size;
        // in.read((char*)&numVerts, sizeof(unsigned int));
        in.read((char*)&numTris, sizeof(unsigned int));
        in.read((char*)&bvh_size, sizeof(unsigned int));
        // std::cout << "num verts = " << numVerts << std::endl;
        std::cout << "num tris  = " << numTris << std::endl;
        std::cout << "num bvhs  = " << bvh_size << std::endl;

        // verts.resize(numVerts);
        // normals.resize(numVerts);
        tris.resize(numTris);
        bvh.resize(bvh_size);

        // in.read((char*)&verts[0], numVerts * sizeof(glm::vec3));
        // in.read((char*)&normals[0], numVerts * sizeof(glm::vec3));
        in.read((char*)&tris[0], numTris * sizeof(Triangle));
        in.read((char*)&bvh[0], bvh_size * sizeof(BVH));

        mesh = CudaMesh(tris, bvh, i);
        // mesh = CudaMesh(verts, normals, tris, bvh, i);

    }

    in.close();

    return list;
}
