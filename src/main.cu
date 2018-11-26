#include "raytracer.cu"

using namespace Progression;

int main(int argc, char* argv[]) {
    auto conf = PG::config::Config(PG_ROOT_DIR "configs/default.toml");
    if (!conf) {
        std::cout << "could not parse config file" << std::endl;
        exit(0);
    }

    PG::EngineInitialize(conf);

    auto scene = Scene::Load("/home/liam/Documents/School/5351/realtime-cuda-raytracer/rayTrace.pgscn");
    auto camera = scene->GetCamera();
    camera->AddComponent<UserCameraComponent>(new UserCameraComponent(camera));

    RayTracer rayTracer;
    int numSpheres = 400;
    rayTracer.Init(1, numSpheres, 5);

    Sphere h_spheres[numSpheres];
    for (int i = 0; i < numSpheres; ++i) {
        // h_spheres[i] = Sphere(make_float3(-6 + 3*i, 0, -10), 1, i);
        float X = rand() / (float) RAND_MAX * 20.f - 10;
        float Y = rand() / (float) RAND_MAX * 20.f - 10;
        float Z = rand() / (float) RAND_MAX * 20.f - 10;
        h_spheres[i] = Sphere(make_float3(X, Y, Z), 1, rand() % 5);
    }

    RTMaterial h_mats[5];
    h_mats[0].kd = make_float3(1, 0, 0);
    h_mats[1].kd = make_float3(0, 1, 0);
    h_mats[2].kd = make_float3(.4, .4, .4);
    h_mats[3].kd = make_float3(1, 1, 0);
    h_mats[4].kd = make_float3(1, 0, 1);
    for (int i = 0; i < 5; ++i) {
        h_mats[i].ks = make_float3(.7, .7, .7);
        h_mats[i].power = 50;
    }

    float3 lights[2];
    lights[0] = normalize(make_float3(0, 0, -1));
    lights[1] = make_float3(1, 1, 1);

    // check(cudaMalloc((void**) &rayTracer.d_spheres, 5 * sizeof(Sphere)));
    check(cudaMemcpy(rayTracer.scene.spheres, h_spheres, sizeof(Sphere) * numSpheres, cudaMemcpyHostToDevice));
    rayTracer.scene.numSpheres = numSpheres;
    check(cudaMemcpy(rayTracer.scene.materials, h_mats, sizeof(RTMaterial) * 5, cudaMemcpyHostToDevice));
    check(cudaMemcpy(rayTracer.scene.lights, lights, 1 * sizeof(float3) * 2, cudaMemcpyHostToDevice));
    rayTracer.scene.numDirectionalLights = 1;
    rayTracer.scene.numPointLights = 0;

    Window::SetRelativeMouse(true);
    PG::Input::PollEvents();
    while (!PG::EngineShutdown) {
        PG::Window::StartFrame();
        PG::Input::PollEvents();

        if (PG::Input::GetKeyDown(PG::PG_K_ESC))
            PG::EngineShutdown = true;

        // scene->Update();
        camera->Update();

        rayTracer.Render(camera);

        PG::Window::EndFrame();
    }

    rayTracer.Free();

    PG::EngineQuit();

    return 0;
}
