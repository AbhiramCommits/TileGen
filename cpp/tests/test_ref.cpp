#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "tilegen/ref.hpp"

using Catch::Approx;

namespace {

tilegen::Csr<double> make_csr(int64_t rows, int64_t cols,
                              std::vector<int64_t> indptr,
                              std::vector<int64_t> indices,
                              std::vector<double> data) {
    tilegen::Csr<double> A;
    A.rows = rows;
    A.cols = cols;
    A.indptr = std::move(indptr);
    A.indices = std::move(indices);
    A.data = std::move(data);
    return A;
}

}  // namespace

TEST_CASE("2x2 identity csr_spmv", "[ref]") {
    // [1 0; 0 1]
    auto A = make_csr(2, 2, {0, 1, 2}, {0, 1}, {1.0, 1.0});
    std::vector<double> x = {3.0, -4.0};
    std::vector<double> y;
    tilegen::csr_spmv(A, x, y);
    REQUIRE(y.size() == 2);
    REQUIRE(y[0] == Approx(3.0));
    REQUIRE(y[1] == Approx(-4.0));
}

TEST_CASE("dense_spmv matches hand computation", "[ref]") {
    // [1 2; 3 4] * [5; 6] = [17; 39]
    std::vector<double> A = {1, 2, 3, 4};
    std::vector<double> x = {5, 6};
    std::vector<double> y;
    tilegen::dense_spmv(A, 2, 2, x, y);
    REQUIRE(y[0] == Approx(17.0));
    REQUIRE(y[1] == Approx(39.0));
}

TEST_CASE("empty row yields zero", "[ref]") {
    // row 1 has no entries
    auto A = make_csr(3, 3, {0, 1, 1, 2}, {0, 2}, {2.0, 5.0});
    std::vector<double> x = {1.0, 1.0, 1.0};
    std::vector<double> y;
    tilegen::csr_spmv(A, x, y);
    REQUIRE(y[0] == Approx(2.0));
    REQUIRE(y[1] == Approx(0.0));
    REQUIRE(y[2] == Approx(5.0));
}

TEST_CASE("all-zero-data matrix yields zero", "[ref]") {
    auto A = make_csr(2, 2, {0, 2, 3}, {0, 1, 1}, {0.0, 0.0, 0.0});
    std::vector<double> x = {7.0, 9.0};
    std::vector<double> y;
    tilegen::csr_spmv(A, x, y);
    REQUIRE(y[0] == Approx(0.0));
    REQUIRE(y[1] == Approx(0.0));
}

TEST_CASE("csr_spmm small", "[ref]") {
    // A = [1 2; 0 3], B = [1 0; 0 1] (K=2) => C = A
    auto A = make_csr(2, 2, {0, 2, 3}, {0, 1, 1}, {1.0, 2.0, 3.0});
    std::vector<double> B = {1, 0, 0, 1};
    std::vector<double> C;
    tilegen::csr_spmm(A, B, 2, C);
    REQUIRE(C.size() == 4);
    REQUIRE(C[0] == Approx(1.0));
    REQUIRE(C[1] == Approx(2.0));
    REQUIRE(C[2] == Approx(0.0));
    REQUIRE(C[3] == Approx(3.0));
}

TEST_CASE("sddmm sampled product", "[ref]") {
    // A = I2 (pattern diag), U = [[1,2],[3,4]], V = [[1,0],[0,1]], K=2
    // UV^T = U. Cvals = A.data * diag(U) = {1, 4}
    auto A = make_csr(2, 2, {0, 1, 2}, {0, 1}, {1.0, 1.0});
    std::vector<double> U = {1, 2, 3, 4};
    std::vector<double> V = {1, 0, 0, 1};
    std::vector<double> C;
    tilegen::sddmm(A, U, V, 2, C);
    REQUIRE(C.size() == 2);
    REQUIRE(C[0] == Approx(1.0));
    REQUIRE(C[1] == Approx(4.0));
}

TEST_CASE("fused_spmv_axpy", "[ref]") {
    // A = [1 1; 0 1], x = [1; 2], w = [1; 1], alpha=2, beta=3
    // A*x = [3; 2]; z = 2*[3;2] + 3*[1;1] = [9; 7]
    auto A = make_csr(2, 2, {0, 2, 3}, {0, 1, 1}, {1.0, 1.0, 1.0});
    std::vector<double> x = {1.0, 2.0};
    std::vector<double> w = {1.0, 1.0};
    std::vector<double> z;
    tilegen::fused_spmv_axpy(A, x, w, 2.0, 3.0, z);
    REQUIRE(z[0] == Approx(9.0));
    REQUIRE(z[1] == Approx(7.0));
}

TEST_CASE("float path produces same result", "[ref]") {
    tilegen::Csr<float> A;
    A.rows = 2;
    A.cols = 2;
    A.indptr = {0, 1, 2};
    A.indices = {0, 1};
    A.data = {1.0f, 1.0f};
    std::vector<float> x = {3.0f, -4.0f};
    std::vector<float> y;
    tilegen::csr_spmv(A, x, y);
    REQUIRE(y[0] == Approx(3.0f));
    REQUIRE(y[1] == Approx(-4.0f));
}
