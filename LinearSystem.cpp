#include "LinearSystem.hpp"
#include <cmath>
#include <algorithm> // for std::swap

// Constructor (builds deep copies of A and b; checks dimensions)
LinearSystem::LinearSystem(const Matrix& A, const Vector& b)
    : mSize(0),
      mpA(nullptr),
      mpb(nullptr)
{
    assert(A.Rows() == A.Cols());
    assert(static_cast<std::size_t>(A.Rows()) == b.Size());
    mSize = A.Rows();
    mpA = new Matrix(A); // deep copy
    mpb = new Vector(b); // deep copy
}

// Destructor
LinearSystem::~LinearSystem() {
    delete mpA;
    delete mpb;
}

// Gaussian elimination with partial pivoting, using only public accessors
Vector LinearSystem::Solve() const {
    const std::size_t n = mSize;

    // 1) Make local copies so we can row‐reduce without altering mpA/mpb
    Matrix A_copy(*mpA);   // deep copy of the n×n matrix
    Vector b_copy(*mpb);   // deep copy of the length‐n vector

    // 2) Forward elimination (one‐based indexing)
    for (std::size_t i = 1; i <= n; ++i) {
        // 2.a) Find pivot in column i (largest absolute value among rows i..n)
        std::size_t pivotRow = i;
        double maxVal = std::abs(A_copy(i, i));
        for (std::size_t r = i + 1; r <= n; ++r) {
            double val = std::abs(A_copy(r, i));
            if (val > maxVal) {
                maxVal = val;
                pivotRow = r;
            }
        }
        assert(maxVal > 1e-14); // matrix must be nonsingular

        // 2.b) If needed, swap row i and pivotRow (both in A_copy and b_copy)
        if (pivotRow != i) {
            for (std::size_t c = 1; c <= n; ++c) {
                std::swap(A_copy(i, c), A_copy(pivotRow, c));
            }
            std::swap(b_copy(i), b_copy(pivotRow));
        }

        // 2.c) Eliminate entries below (i,i)
        double pivotVal = A_copy(i, i);
        for (std::size_t r = i + 1; r <= n; ++r) {
            double factor = A_copy(r, i) / pivotVal;
            // update row r in A_copy
            for (std::size_t c = i; c <= n; ++c) {
                A_copy(r, c) -= factor * A_copy(i, c);
            }
            // update RHS
            b_copy(r) -= factor * b_copy(i);
        }
    }

    // 3) Back substitution (one‐based indexing)
    Vector x(n); // final solution vector, initially all zeros
    for (std::size_t ii = n; ii >= 1; --ii) {
        double sum = b_copy(ii);
        for (std::size_t j = ii + 1; j <= n; ++j) {
            sum -= A_copy(ii, j) * x(j);
        }
        x(ii) = sum / A_copy(ii, ii);
        if (ii == 1) break; // prevent underflow of size_t
    }

    return x;
}
