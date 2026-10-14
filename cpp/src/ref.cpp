#include "tilegen/ref.hpp"

namespace tilegen {

template <typename T>
void csr_spmv(const Csr<T>& A, const std::vector<T>& x, std::vector<T>& y) {
    y.assign(static_cast<size_t>(A.rows), T(0));
    for (int64_t i = 0; i < A.rows; ++i) {
        T acc = T(0);
        for (int64_t p = A.indptr[i]; p < A.indptr[i + 1]; ++p) {
            acc += A.data[p] * x[A.indices[p]];
        }
        y[i] = acc;
    }
}

template <typename T>
void dense_spmv(const std::vector<T>& A, int64_t m, int64_t n,
                const std::vector<T>& x, std::vector<T>& y) {
    y.assign(static_cast<size_t>(m), T(0));
    for (int64_t i = 0; i < m; ++i) {
        T acc = T(0);
        for (int64_t j = 0; j < n; ++j) {
            acc += A[i * n + j] * x[j];
        }
        y[i] = acc;
    }
}

template <typename T>
void csr_spmm(const Csr<T>& A, const std::vector<T>& B, int64_t K,
              std::vector<T>& C) {
    C.assign(static_cast<size_t>(A.rows) * static_cast<size_t>(K), T(0));
    for (int64_t i = 0; i < A.rows; ++i) {
        for (int64_t p = A.indptr[i]; p < A.indptr[i + 1]; ++p) {
            int64_t j = A.indices[p];
            T a = A.data[p];
            for (int64_t k = 0; k < K; ++k) {
                C[i * K + k] += a * B[j * K + k];
            }
        }
    }
}

template <typename T>
void sddmm(const Csr<T>& A, const std::vector<T>& U, const std::vector<T>& V,
           int64_t K, std::vector<T>& Cvals) {
    Cvals.assign(static_cast<size_t>(A.data.size()), T(0));
    for (int64_t i = 0; i < A.rows; ++i) {
        for (int64_t p = A.indptr[i]; p < A.indptr[i + 1]; ++p) {
            int64_t j = A.indices[p];
            T acc = T(0);
            for (int64_t k = 0; k < K; ++k) {
                acc += U[i * K + k] * V[j * K + k];
            }
            Cvals[p] = A.data[p] * acc;
        }
    }
}

template <typename T>
void fused_spmv_axpy(const Csr<T>& A, const std::vector<T>& x,
                     const std::vector<T>& w, T alpha, T beta,
                     std::vector<T>& z) {
    z.assign(static_cast<size_t>(A.rows), T(0));
    for (int64_t i = 0; i < A.rows; ++i) {
        T acc = T(0);
        for (int64_t p = A.indptr[i]; p < A.indptr[i + 1]; ++p) {
            acc += A.data[p] * x[A.indices[p]];
        }
        z[i] = alpha * acc + beta * w[i];
    }
}

template void csr_spmv<double>(const Csr<double>&, const std::vector<double>&,
                               std::vector<double>&);
template void csr_spmv<float>(const Csr<float>&, const std::vector<float>&,
                              std::vector<float>&);
template void dense_spmv<double>(const std::vector<double>&, int64_t, int64_t,
                                 const std::vector<double>&, std::vector<double>&);
template void dense_spmv<float>(const std::vector<float>&, int64_t, int64_t,
                                const std::vector<float>&, std::vector<float>&);
template void csr_spmm<double>(const Csr<double>&, const std::vector<double>&,
                               int64_t, std::vector<double>&);
template void csr_spmm<float>(const Csr<float>&, const std::vector<float>&,
                              int64_t, std::vector<float>&);
template void sddmm<double>(const Csr<double>&, const std::vector<double>&,
                            const std::vector<double>&, int64_t, std::vector<double>&);
template void sddmm<float>(const Csr<float>&, const std::vector<float>&,
                           const std::vector<float>&, int64_t, std::vector<float>&);
template void fused_spmv_axpy<double>(const Csr<double>&, const std::vector<double>&,
                                      const std::vector<double>&, double, double,
                                      std::vector<double>&);
template void fused_spmv_axpy<float>(const Csr<float>&, const std::vector<float>&,
                                     const std::vector<float>&, float, float,
                                     std::vector<float>&);

}  // namespace tilegen
