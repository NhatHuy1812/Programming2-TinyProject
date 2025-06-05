#include "PosSymLinSystem.hpp"
#include <cmath>
#include <cassert>

// Helper: Euclidean norm of a vector
static double Norm(const Vector& v) {
    return std::sqrt(v.Dot(v));
}

Vector PosSymLinSystem::Solve() const {
    const std::size_t n = mSize;

    // 1) Verify symmetry: A(i,j) == A(j,i) for all 1 ≤ i < j ≤ n
    for (std::size_t i = 1; i <= n; ++i) {
        for (std::size_t j = i + 1; j <= n; ++j) {
            double aij = (*mpA)(i, j);
            double aji = (*mpA)(j, i);
            assert(std::abs(aij - aji) < 1e-12);
        }
    }

    // 2) Conjugate Gradient (initial guess x₀ = 0)
    Vector x(n);             // initially zero vector
    Vector r = *mpb;         // r₀ = b − A*x₀ = b
    Vector p = r;            // p₀ = r₀
    double r_norm_sq_old = r.Dot(r);

    // If b is (almost) zero, solution is x=0
    if (std::sqrt(r_norm_sq_old) < 1e-14) {
        return x;
    }

    const double tol = 1e-10;
    for (std::size_t k = 0; k < n; ++k) {
        // Compute A*p (matrix‐vector product)
        Vector Ap = (*mpA) * p;

        double alpha = r_norm_sq_old / p.Dot(Ap);

        // x_{k+1} = x_k + α p_k
        x = x + (alpha * p);

        // r_{k+1} = r_k − α Ap
        r = r - (alpha * Ap);

        double r_norm_sq_new = r.Dot(r);
        if (std::sqrt(r_norm_sq_new) < tol) {
            break;
        }

        double beta = r_norm_sq_new / r_norm_sq_old;
        // p_{k+1} = r_{k+1} + β p_k
        p = r + (beta * p);

        r_norm_sq_old = r_norm_sq_new;
    }
    return x;
}
