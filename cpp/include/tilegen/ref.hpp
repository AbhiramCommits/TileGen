// TileGen reference kernels -- plainly written, obviously correct, no
// cleverness.  Templated on the scalar type so both float and double are
// available; explicit instantiations for double and float live in ref.cpp.
#pragma once

#include <cstdint>
#include <vector>

namespace tilegen {

template <typename T>
struct Csr {
    int64_t rows = 0;
    int64_t cols = 0;
    std::vector<int64_t> indptr;
    std::vector<int64_t> indices;
    std::vector<T> data;
};

// y = A * x  (A sparse CSR, x dense) -> y dense
template <typename T>
void csr_spmv(const Csr<T>& A, const std::vector<T>& x, std::vector<T>& y);

// y = A * x  (A dense row-major m x n)
template <typename T>
void dense_spmv(const std::vector<T>& A, int64_t m, int64_t n,
                const std::vector<T>& x, std::vector<T>& y);

// C = A * B  (A sparse CSR m x n, B dense n x K row-major, C dense m x K)
template <typename T>
void csr_spmm(const Csr<T>& A, const std::vector<T>& B, int64_t K,
              std::vector<T>& C);

// C[i,j] = A[i,j] * sum_k U[i,k] * V[j,k]  (sampled; C follows A's pattern)
template <typename T>
void sddmm(const Csr<T>& A, const std::vector<T>& U, const std::vector<T>& V,
           int64_t K, std::vector<T>& Cvals);

// z = alpha * (A * x) + beta * w
template <typename T>
void fused_spmv_axpy(const Csr<T>& A, const std::vector<T>& x,
                     const std::vector<T>& w, T alpha, T beta,
                     std::vector<T>& z);

extern template void csr_spmv<double>(const Csr<double>&, const std::vector<double>&,
                                      std::vector<double>&);
extern template void csr_spmv<float>(const Csr<float>&, const std::vector<float>&,
                                     std::vector<float>&);
extern template void dense_spmv<double>(const std::vector<double>&, int64_t, int64_t,
                                        const std::vector<double>&, std::vector<double>&);
extern template void dense_spmv<float>(const std::vector<float>&, int64_t, int64_t,
                                       const std::vector<float>&, std::vector<float>&);
extern template void csr_spmm<double>(const Csr<double>&, const std::vector<double>&,
                                      int64_t, std::vector<double>&);
extern template void csr_spmm<float>(const Csr<float>&, const std::vector<float>&,
                                     int64_t, std::vector<float>&);
extern template void sddmm<double>(const Csr<double>&, const std::vector<double>&,
                                   const std::vector<double>&, int64_t, std::vector<double>&);
extern template void sddmm<float>(const Csr<float>&, const std::vector<float>&,
                                  const std::vector<float>&, int64_t, std::vector<float>&);
extern template void fused_spmv_axpy<double>(const Csr<double>&, const std::vector<double>&,
                                             const std::vector<double>&, double, double,
                                             std::vector<double>&);
extern template void fused_spmv_axpy<float>(const Csr<float>&, const std::vector<float>&,
                                            const std::vector<float>&, float, float,
                                            std::vector<float>&);

}  // namespace tilegen
