// FastShadowGPU: annual sunlit fraction of a DSM tile, computed on the GPU with
// the other tiles in the folder as shadow-casting context.
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <tiffio.h>

namespace fs = std::filesystem;

std::vector<float> sunlit_fraction(const std::vector<float>& dsm, int gw, int gh, int x0, int y0, int tw, int th,
                                   float cell, const std::vector<std::pair<float, float>>& sun); // shadow.cu

[[noreturn]] static void die(const std::string& msg)
{
    fprintf(stderr, "%s\n", msg.c_str());
    exit(1);
}

// ---------------------------------------------------------------- GeoTIFF I/O

// GeoTIFF tags, which libtiff does not know by itself.
enum : uint32_t { kPixelScale = 33550, kTiepoint = 33922, kGeoKeys = 34735, kGeoDoubles = 34736, kGeoAscii = 34737 };

static TIFFExtendProc g_parentExtender = nullptr;

static void geotiff_tags(TIFF* tif)
{
    static const TIFFFieldInfo info[] = {
        { kPixelScale, -1, -1, TIFF_DOUBLE, FIELD_CUSTOM, 1, 1, (char*)"ModelPixelScaleTag" },
        { kTiepoint,   -1, -1, TIFF_DOUBLE, FIELD_CUSTOM, 1, 1, (char*)"ModelTiepointTag" },
        { kGeoKeys,    -1, -1, TIFF_SHORT,  FIELD_CUSTOM, 1, 1, (char*)"GeoKeyDirectoryTag" },
        { kGeoDoubles, -1, -1, TIFF_DOUBLE, FIELD_CUSTOM, 1, 1, (char*)"GeoDoubleParamsTag" },
        { kGeoAscii,   -1, -1, TIFF_ASCII,  FIELD_CUSTOM, 1, 0, (char*)"GeoAsciiParamsTag" },
    };
    TIFFMergeFieldInfo(tif, info, sizeof(info) / sizeof(info[0]));
    if (g_parentExtender) g_parentExtender(tif);
}

// Single-band float raster with the georeferencing needed to place and write it.
struct Raster {
    uint32_t w = 0, h = 0;
    std::vector<float> z;
    std::vector<double> scale, tie, geoDoubles; // tie[3], tie[4]: map x, y of the top-left corner
    std::vector<uint16_t> geoKeys;
    std::string geoAscii;
};

template <class T>
static std::vector<T> get_array(TIFF* tif, uint32_t tag)
{
    uint16_t n = 0;
    T* p = nullptr;
    return TIFFGetField(tif, tag, &n, &p) && p ? std::vector<T>(p, p + n) : std::vector<T>();
}

static Raster read_tiff(const fs::path& path)
{
    const std::string name = path.string();
    TIFF* tif = TIFFOpen(name.c_str(), "r");
    if (!tif) die("Cannot open " + name);

    Raster r;
    uint16_t bits = 0, format = 0, bands = 0;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &r.w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &r.h);
    TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &bits);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &format);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &bands);
    if (bits != 32 || format != SAMPLEFORMAT_IEEEFP || bands != 1) die(name + ": expected a single-band float32 GeoTIFF");

    r.scale = get_array<double>(tif, kPixelScale);
    r.tie = get_array<double>(tif, kTiepoint);
    r.geoKeys = get_array<uint16_t>(tif, kGeoKeys);
    r.geoDoubles = get_array<double>(tif, kGeoDoubles);
    char* ascii = nullptr;
    if (TIFFGetField(tif, kGeoAscii, &ascii) && ascii) r.geoAscii = ascii;
    if (r.scale.size() < 2 || r.tie.size() < 6) die(name + ": missing GeoTIFF pixel scale or tiepoint");

    r.z.resize((size_t)r.w * r.h);
    if (TIFFIsTiled(tif)) {
        uint32_t bw = 0, bh = 0;
        TIFFGetField(tif, TIFFTAG_TILEWIDTH, &bw);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &bh);
        std::vector<float> buf((size_t)bw * bh);
        for (uint32_t y = 0; y < r.h; y += bh)
            for (uint32_t x = 0; x < r.w; x += bw) {
                if (TIFFReadTile(tif, buf.data(), x, y, 0, 0) < 0) die("Read error in " + name);
                for (uint32_t k = 0; k < bh && y + k < r.h; ++k)
                    std::copy_n(&buf[(size_t)k * bw], std::min(bw, r.w - x), &r.z[(size_t)(y + k) * r.w + x]);
            }
    } else {
        for (uint32_t y = 0; y < r.h; ++y)
            if (TIFFReadScanline(tif, &r.z[(size_t)y * r.w], y) < 0) die("Read error in " + name);
    }
    TIFFClose(tif);
    return r;
}

static void write_tiff(const fs::path& path, const Raster& r)
{
    const std::string name = path.string();
    TIFF* tif = TIFFOpen(name.c_str(), "w");
    if (!tif) die("Cannot create " + name);

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, r.w);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, r.h);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_ADOBE_DEFLATE);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, 16);
    TIFFSetField(tif, kPixelScale, (uint16_t)r.scale.size(), r.scale.data());
    TIFFSetField(tif, kTiepoint, (uint16_t)r.tie.size(), r.tie.data());
    if (!r.geoKeys.empty()) TIFFSetField(tif, kGeoKeys, (uint16_t)r.geoKeys.size(), r.geoKeys.data());
    if (!r.geoDoubles.empty()) TIFFSetField(tif, kGeoDoubles, (uint16_t)r.geoDoubles.size(), r.geoDoubles.data());
    if (!r.geoAscii.empty()) TIFFSetField(tif, kGeoAscii, r.geoAscii.c_str());

    for (uint32_t y = 0; y < r.h; ++y)
        if (TIFFWriteScanline(tif, (void*)&r.z[(size_t)y * r.w], y) < 0) die("Write error in " + name);
    TIFFClose(tif);
}

// --------------------------------------------------------------- Sun position

// Hourly sun (azimuth from north clockwise, altitude) in degrees over one year in UTC,
// with the PSA algorithm (Blanco-Muriel et al., Solar Energy 70(5), 431-441, 2001).
static std::vector<std::pair<float, float>> sun_positions(double lon, double lat, int year)
{
    const double pi = 3.14159265358979323846, rad = pi / 180.0;
    const long a = (1 - 14) / 12; // Julian day number of 1 January
    const long jdn = (1461 * (year + 4800 + a)) / 4 + (367 * (1 - 2 - 12 * a)) / 12 - (3 * ((year + 4900 + a) / 100)) / 4 + 1 - 32075;
    const int days = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 ? 366 : 365;
    const double phi = lat * rad;

    std::vector<std::pair<float, float>> sun;
    for (int h = 0; h < days * 24; ++h) {
        const double n = jdn - 0.5 + h / 24.0 - 2451545.0; // days since J2000.0
        const double omega = 2.1429 - 0.0010394594 * n;
        const double L = 4.8950630 + 0.017202791698 * n;
        const double g = 6.2400600 + 0.0172019699 * n;
        const double lambda = L + 0.03341607 * sin(g) + 0.00034894 * sin(2 * g) - 0.0001134 - 0.0000203 * sin(omega);
        const double eps = 0.4090928 - 6.2140e-9 * n + 0.0000396 * cos(omega);
        double ra = atan2(cos(eps) * sin(lambda), cos(lambda));
        if (ra < 0) ra += 2 * pi;
        const double dec = asin(sin(eps) * sin(lambda));
        const double gmst = 6.6974243242 + 0.0657098283 * n + h % 24;
        const double ha = (gmst * 15 + lon) * rad - ra;
        double zen = acos(cos(phi) * cos(ha) * cos(dec) + sin(dec) * sin(phi));
        double az = atan2(-sin(ha), tan(dec) * cos(phi) - sin(phi) * cos(ha));
        if (az < 0) az += 2 * pi;
        zen += 6371.01 / 149597890.0 * sin(zen); // parallax
        sun.emplace_back((float)(az / rad), (float)(90.0 - zen / rad));
    }
    return sun;
}

// ----------------------------------------------------------------------- Main

int main(int argc, char** argv)
{
    if (argc < 5) {
        printf("Usage: FastShadowGPU <tiles_dir> <target_tile> <lon> <lat> [output.tif]\n");
        return 1;
    }
    const fs::path dir = argv[1];
    const fs::path target = fs::path(argv[2]).filename();
    const float lon = std::stof(argv[3]), lat = std::stof(argv[4]);
    const fs::path out = argc > 5 ? fs::path(argv[5]) : fs::path(target.stem().string() + "_sunlit.tif");
    g_parentExtender = TIFFSetTagExtender(geotiff_tags);

    // Every GeoTIFF in the folder is part of the shadow-casting context.
    if (!fs::is_directory(dir)) die("Not a folder: " + dir.string());
    std::vector<fs::path> paths;
    for (const auto& e : fs::directory_iterator(dir)) {
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".tif" || ext == ".tiff") paths.push_back(e.path());
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) die("No GeoTIFFs in " + dir.string());

    std::vector<Raster> tiles;
    for (const auto& p : paths) tiles.push_back(read_tiff(p));

    // Mosaic all tiles on one grid (same square pixel size required).
    const double cell = tiles[0].scale[0];
    double minX = 1e300, maxX = -1e300, minY = 1e300, maxY = -1e300;
    for (const Raster& t : tiles) {
        if (std::abs(t.scale[0] - cell) > 1e-6 || std::abs(t.scale[1] - cell) > 1e-6)
            die("All tiles need the same square pixel size");
        minX = std::min(minX, t.tie[3]);
        maxX = std::max(maxX, t.tie[3] + t.w * cell);
        maxY = std::max(maxY, t.tie[4]);
        minY = std::min(minY, t.tie[4] - t.h * cell);
    }
    const int gw = (int)std::lround((maxX - minX) / cell), gh = (int)std::lround((maxY - minY) / cell);
    std::vector<float> dsm((size_t)gw * gh, 0.0f);

    int tgt = -1, tx = 0, ty = 0;
    for (size_t k = 0; k < tiles.size(); ++k) {
        Raster& t = tiles[k];
        const int ox = (int)std::lround((t.tie[3] - minX) / cell), oy = (int)std::lround((maxY - t.tie[4]) / cell);
        const int w = std::min((int)t.w, gw - ox), h = std::min((int)t.h, gh - oy);
        for (int y = 0; y < h; ++y)
            std::copy_n(&t.z[(size_t)y * t.w], w, &dsm[(size_t)(oy + y) * gw + ox]);
        t.z = {};
        if (paths[k].filename() == target) { tgt = (int)k; tx = ox; ty = oy; }
    }
    if (tgt < 0) die("Target tile " + target.string() + " is not in " + dir.string());
    Raster& t = tiles[tgt];
    printf("%zu tiles, mosaic %d x %d px, pixel %g m; target %s (%u x %u) at (%d, %d)\n",
           tiles.size(), gw, gh, cell, target.string().c_str(), t.w, t.h, tx, ty);

    const auto t0 = std::chrono::steady_clock::now();
    t.z = sunlit_fraction(dsm, gw, gh, tx, ty, (int)t.w, (int)t.h, (float)cell, sun_positions(lon, lat, 2026));
    const auto t1 = std::chrono::steady_clock::now();
    printf("Shadowing: %.2f s\n", std::chrono::duration<double>(t1 - t0).count());

    write_tiff(out, t);
    printf("Wrote %s\n", out.string().c_str());
    return 0;
}
