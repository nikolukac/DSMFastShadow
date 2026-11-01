// GPU part: annual sunlit fraction of a target window inside a larger DSM.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t e = (call);                                                   \
        if (e != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e),    \
                    __FILE__, __LINE__);                                          \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

constexpr float kSectorDeg = 5.0f;     // sky sector size (azimuth and altitude)
constexpr float kMinAltitudeDeg = 1.0f; // lower sun counts as night
constexpr float kHitEps = 0.01f;       // m, a ray is blocked if the surface is this much above it

// One thread per target pixel: march a ray towards direction d over the DSM
// (2D DDA in cell units, heights in metres) and add `hours` if it is blocked.
__global__ void trace(const float* __restrict__ dsm, int gw, int gh, int x0, int y0, int tw, int th,
                      float3 d, float cell, float zmax, float hours, float* __restrict__ shaded)
{
    const int lx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ly = blockIdx.y * blockDim.y + threadIdx.y;
    if (lx >= tw || ly >= th) return;
    const int x = x0 + lx, y = y0 + ly;

    // Start two cells along the ray to avoid self-shadowing on the pixel's own surface.
    const float ox = x + 0.5f + 2.0f * d.x;
    const float oy = y + 0.5f + 2.0f * d.y;
    const float dz = d.z * cell; // metres climbed per cell of path
    const float oz = dsm[(size_t)y * gw + x] + 2.0f * dz;

    int ix = (int)floorf(ox), iy = (int)floorf(oy);
    const int sx = d.x >= 0.0f ? 1 : -1, sy = d.y >= 0.0f ? 1 : -1;
    const float dtx = 1.0f / fmaxf(fabsf(d.x), 1e-8f), dty = 1.0f / fmaxf(fabsf(d.y), 1e-8f);
    float tx = (sx > 0 ? ix + 1.0f - ox : ox - ix) * dtx;
    float ty = (sy > 0 ? iy + 1.0f - oy : oy - iy) * dty;

    for (;;) {
        float t;
        if (tx < ty) { t = tx; tx += dtx; ix += sx; }
        else         { t = ty; ty += dty; iy += sy; }

        const float z = oz + dz * t;
        if ((unsigned)ix >= (unsigned)gw || (unsigned)iy >= (unsigned)gh || z > zmax) return;
        if (dsm[(size_t)iy * gw + ix] - z > kHitEps) {
            shaded[ly * tw + lx] += hours;
            return;
        }
    }
}

// dsm: gw x gh heights (m), row-major, north up. Target window: (x0, y0, tw, th).
// sun: hourly (azimuth from north clockwise, altitude) in degrees.
// Returns, per target pixel, the fraction of daytime hours in direct sun.
std::vector<float> sunlit_fraction(const std::vector<float>& dsm, int gw, int gh, int x0, int y0, int tw, int th,
                                   float cell, const std::vector<std::pair<float, float>>& sun)
{
    // Bin the daytime hours into sky sectors; each sector is traced once, from its centre.
    const int nAz = (int)(360.0f / kSectorDeg), nAlt = (int)(90.0f / kSectorDeg);
    std::vector<float> hours(nAz * nAlt, 0.0f);
    float daytime = 0.0f;
    for (const auto& [az, alt] : sun) {
        if (alt < kMinAltitudeDeg) continue;
        int i = (int)(az / kSectorDeg) % nAz, j = std::min((int)(alt / kSectorDeg), nAlt - 1);
        hours[i * nAlt + j] += 1.0f;
        daytime += 1.0f;
    }

    float zmax = dsm[0];
    for (float z : dsm) zmax = std::max(zmax, z);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);

    const size_t nGrid = (size_t)gw * gh, nTarget = (size_t)tw * th;
    float *dsm_d, *shaded_d;
    CUDA_CHECK(cudaMalloc(&dsm_d, nGrid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&shaded_d, nTarget * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dsm_d, dsm.data(), nGrid * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(shaded_d, 0, nTarget * sizeof(float)));

    const dim3 block(16, 16), grid((tw + 15) / 16, (th + 15) / 16);
    const float rad = 3.14159265f / 180.0f;
    int sectors = 0;
    for (int i = 0; i < nAz; ++i)
        for (int j = 0; j < nAlt; ++j) {
            if (hours[i * nAlt + j] == 0.0f) continue;
            const float az = (i + 0.5f) * kSectorDeg * rad, alt = (j + 0.5f) * kSectorDeg * rad;
            const float3 d = make_float3(cosf(alt) * sinf(az), -cosf(alt) * cosf(az), sinf(alt)); // x east, y south
            trace<<<grid, block>>>(dsm_d, gw, gh, x0, y0, tw, th, d, cell, zmax, hours[i * nAlt + j], shaded_d);
            CUDA_CHECK(cudaGetLastError());
            ++sectors;
        }
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Traced %d sky sectors (%g daytime hours)\n", sectors, daytime);

    std::vector<float> lit(nTarget);
    CUDA_CHECK(cudaMemcpy(lit.data(), shaded_d, nTarget * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dsm_d));
    CUDA_CHECK(cudaFree(shaded_d));
    for (float& v : lit) v = daytime > 0.0f ? 1.0f - v / daytime : 0.0f;
    return lit;
}
