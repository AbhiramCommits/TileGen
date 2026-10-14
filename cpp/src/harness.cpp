// TileGen host harness.
//
// Loads a CSR matrix from the little-endian .bin dump produced by
// tilegen.matrices, dlopen()s a generated kernel exposing the
// extern "C" void launch_<name>(...) ABI, runs it, times it with std::chrono
// (or cudaEvent when built with CUDA), checks it against the reference kernel,
// and prints a JSON result line.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <dlfcn.h>

#include "tilegen/ref.hpp"

#ifdef TILEGEN_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace {

struct Matrix {
    int64_t rows = 0, cols = 0, nnz = 0;
    std::vector<int64_t> indptr, indices;
    std::vector<double> data;
};

Matrix load_bin(const std::string& path) {
    std::ifstream fh(path, std::ios::binary);
    if (!fh) {
        std::cerr << "error: cannot open matrix " << path << "\n";
        std::exit(2);
    }
    uint32_t magic = 0;
    fh.read(reinterpret_cast<char*>(&magic), 4);
    Matrix m;
    fh.read(reinterpret_cast<char*>(&m.rows), 8);
    fh.read(reinterpret_cast<char*>(&m.cols), 8);
    fh.read(reinterpret_cast<char*>(&m.nnz), 8);
    m.indptr.resize(m.rows + 1);
    m.indices.resize(m.nnz);
    m.data.resize(m.nnz);
    fh.read(reinterpret_cast<char*>(m.indptr.data()), 8 * (m.rows + 1));
    fh.read(reinterpret_cast<char*>(m.indices.data()), 8 * m.nnz);
    fh.read(reinterpret_cast<char*>(m.data.data()), 8 * m.nnz);
    return m;
}

tilegen::Csr<double> to_ref(const Matrix& m) {
    tilegen::Csr<double> A;
    A.rows = m.rows;
    A.cols = m.cols;
    A.indptr = m.indptr;
    A.indices = m.indices;
    A.data = m.data;
    return A;
}

struct Args {
    std::string kernel;
    std::string matrix;
    std::string lib;
    std::string config = "{}";
    std::string backend = "cpu";
    int iters = 20;
    int warmup = 5;
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "error: missing value for " << k << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (k == "--kernel") a.kernel = next();
        else if (k == "--matrix") a.matrix = next();
        else if (k == "--lib") a.lib = next();
        else if (k == "--config") a.config = next();
        else if (k == "--backend") a.backend = next();
        else if (k == "--iters") a.iters = std::atoi(next().c_str());
        else if (k == "--warmup") a.warmup = std::atoi(next().c_str());
        else { std::cerr << "error: unknown flag " << k << "\n"; std::exit(2); }
    }
    return a;
}

double now_ms() {
    using namespace std::chrono;
    return duration_cast<duration<double, std::milli>>(
               steady_clock::now().time_since_epoch()).count();
}

std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse(argc, argv);
    if (args.kernel.empty() || args.matrix.empty() || args.lib.empty()) {
        std::cerr << "usage: tilegen_bench --kernel K --matrix M.bin --lib L.so "
                     "[--config JSON] [--backend cpu|cuda] [--iters N]\n";
        return 2;
    }

    Matrix m = load_bin(args.matrix);
    auto A = to_ref(m);
    int64_t M = m.rows, N = m.cols, nnz = m.nnz;
    int64_t K = 32;

    void* handle = dlopen(args.lib.c_str(), RTLD_NOW);
    if (!handle) {
        std::cerr << "error: dlopen failed: " << dlerror() << "\n";
        return 3;
    }
    std::string sym = "launch_" + args.kernel;
    void* fn = dlsym(handle, sym.c_str());
    if (!fn) {
        std::cerr << "error: symbol " << sym << " not found: " << dlerror() << "\n";
        return 3;
    }

    // Inputs / outputs sized for the largest case.
    std::vector<double> x(N), w(M), y(M), z(M);
    std::vector<double> B(static_cast<size_t>(N) * K), U(static_cast<size_t>(M) * K),
        V(static_cast<size_t>(N) * K), C(static_cast<size_t>(M) * K),
        Cvals(nnz);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.5 + 0.001 * static_cast<double>(i % 97);
    for (size_t i = 0; i < w.size(); ++i) w[i] = -0.25 + 0.002 * static_cast<double>(i % 53);
    for (size_t i = 0; i < B.size(); ++i) B[i] = 0.1 * static_cast<double>((i % 31) - 15);
    for (size_t i = 0; i < U.size(); ++i) U[i] = 0.05 * static_cast<double>((i % 23) - 11);
    for (size_t i = 0; i < V.size(); ++i) V[i] = -0.03 * static_cast<double>((i % 19) - 9);
    const double alpha = 0.7, beta = -1.3;

    // reference + flop/byte model
    double flops = 0.0, bytes = 0.0;
    std::vector<double> ref;
    if (args.kernel == "spmv") {
        tilegen::csr_spmv(A, x, y);
        ref = y;
        flops = 2.0 * nnz;
        bytes = nnz * 16.0 + (M + 1) * 8.0 + nnz * 8.0 + M * 8.0;
    } else if (args.kernel == "spmm") {
        tilegen::csr_spmm(A, B, K, C);
        ref = C;
        flops = 2.0 * nnz * K;
        bytes = nnz * 16.0 + (M + 1) * 8.0 + nnz * K * 8.0 + M * K * 8.0;
    } else if (args.kernel == "sddmm") {
        tilegen::sddmm(A, U, V, K, Cvals);
        ref = Cvals;
        flops = 2.0 * nnz * K;
        bytes = nnz * 16.0 + (M + 1) * 8.0 + 2.0 * nnz * K * 8.0 + nnz * 8.0;
    } else if (args.kernel == "fused_spmv_axpy") {
        tilegen::fused_spmv_axpy(A, x, w, alpha, beta, z);
        ref = z;
        flops = 2.0 * nnz;
        bytes = nnz * 16.0 + (M + 1) * 8.0 + nnz * 8.0 + 2 * M * 8.0;
    } else if (args.kernel == "axpy") {
        ref.resize(M);
        for (int64_t i = 0; i < M; ++i) ref[i] = alpha * x[i] + beta * w[i];
        flops = 2.0 * M;
        bytes = 3.0 * M * 8.0;
    } else {
        std::cerr << "error: unknown kernel " << args.kernel << "\n";
        return 2;
    }

    auto run_once = [&]() {
        if (args.kernel == "spmv") {
            using F = void (*)(const int64_t*, const int64_t*, const double*,
                               const double*, double*, int64_t, int64_t, int64_t);
            reinterpret_cast<F>(fn)(m.indptr.data(), m.indices.data(), m.data.data(),
                                    x.data(), y.data(), M, N, nnz);
        } else if (args.kernel == "spmm") {
            using F = void (*)(const int64_t*, const int64_t*, const double*,
                               const double*, double*, int64_t, int64_t, int64_t, int64_t);
            std::fill(C.begin(), C.end(), 0.0);
            reinterpret_cast<F>(fn)(m.indptr.data(), m.indices.data(), m.data.data(),
                                    B.data(), C.data(), M, N, K, nnz);
        } else if (args.kernel == "sddmm") {
            using F = void (*)(const int64_t*, const int64_t*, const double*,
                               const double*, const double*, double*, int64_t, int64_t,
                               int64_t, int64_t);
            reinterpret_cast<F>(fn)(m.indptr.data(), m.indices.data(), m.data.data(),
                                    U.data(), V.data(), Cvals.data(), M, N, K, nnz);
        } else if (args.kernel == "fused_spmv_axpy") {
            using F = void (*)(const int64_t*, const int64_t*, const double*,
                               const double*, const double*, double*, double, double,
                               int64_t, int64_t, int64_t);
            reinterpret_cast<F>(fn)(m.indptr.data(), m.indices.data(), m.data.data(),
                                    x.data(), w.data(), z.data(), alpha, beta, M, N, nnz);
        } else if (args.kernel == "axpy") {
            using F = void (*)(const double*, const double*, double*, double, double,
                               int64_t);
            reinterpret_cast<F>(fn)(x.data(), w.data(), z.data(), alpha, beta, M);
        }
    };

    for (int i = 0; i < args.warmup; ++i) run_once();

    std::vector<double> times;
    double t0 = now_ms();
    for (int i = 0; i < args.iters; ++i) {
        double a = now_ms();
        run_once();
        double b = now_ms();
        times.push_back(b - a);
    }
    (void)t0;

    double sum = 0.0, mn = times.empty() ? 0.0 : times[0];
    for (double t : times) { sum += t; if (t < mn) mn = t; }
    double mean = times.empty() ? 0.0 : sum / times.size();

    // error vs reference
    const std::vector<double>& out = (args.kernel == "spmm") ? C
        : (args.kernel == "sddmm") ? Cvals
        : (args.kernel == "fused_spmv_axpy") ? z
        : (args.kernel == "axpy") ? z : y;
    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < out.size() && i < ref.size(); ++i) {
        double d = std::fabs(out[i] - ref[i]);
        if (d > max_abs) max_abs = d;
        double denom = std::fabs(ref[i]) > 1e-300 ? std::fabs(ref[i]) : 1e-300;
        double r = d / denom;
        if (r > max_rel) max_rel = r;
    }

    double gflops = (mn > 0.0) ? flops / (mn * 1e6) : 0.0;
    double bw = (mn > 0.0) ? bytes / (mn * 1e6) : 0.0;  // ms -> GB/s

    std::ostringstream js;
    js << "{";
    js << "\"kernel\":\"" << json_escape(args.kernel) << "\",";
    js << "\"config\":" << args.config << ",";
    js << "\"backend\":\"" << json_escape(args.backend) << "\",";
    js << "\"matrix\":\"" << json_escape(args.matrix) << "\",";
    js << "\"rows\":" << M << ",\"cols\":" << N << ",\"nnz\":" << nnz << ",";
    js << "\"ms_mean\":" << mean << ",";
    js << "\"ms_min\":" << mn << ",";
    js << "\"gflops\":" << gflops << ",";
    js << "\"bandwidth_gbs\":" << bw << ",";
    js << "\"max_abs_err\":" << max_abs << ",";
    js << "\"max_rel_err\":" << max_rel;
    js << "}";
    std::cout << js.str() << std::endl;
    dlclose(handle);
    return 0;
}
