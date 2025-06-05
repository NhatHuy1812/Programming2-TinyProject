#include "Matrix.hpp"
#include <iomanip>
#include <algorithm> // for std::swap, std::fill
#include <cmath>

// ---------------------- Constructor & Destructor ----------------------

// Allocate rows × cols, initialize all entries to zero
Matrix::Matrix(std::size_t rows, std::size_t cols)
    : mNumRows(rows),
      mNumCols(cols),
      mData(nullptr)
{
    assert(rows > 0 && cols > 0);
    mData = new double*[mNumRows];
    for (std::size_t i = 0; i < mNumRows; ++i) {
        mData[i] = new double[mNumCols];
        std::fill(mData[i], mData[i] + mNumCols, 0.0);
    }
}

// Copy constructor (deep copy)
Matrix::Matrix(const Matrix& other)
    : mNumRows(other.mNumRows),
      mNumCols(other.mNumCols),
      mData(nullptr)
{
    mData = new double*[mNumRows];
    for (std::size_t i = 0; i < mNumRows; ++i) {
        mData[i] = new double[mNumCols];
        std::copy(other.mData[i], other.mData[i] + mNumCols, mData[i]);
    }
}

// Copy assignment (deep copy; handle self‐assignment)
Matrix& Matrix::operator=(const Matrix& other) {
    if (this == &other) {
        return *this;
    }
    if (mNumRows != other.mNumRows || mNumCols != other.mNumCols) {
        // delete old data
        for (std::size_t i = 0; i < mNumRows; ++i) {
            delete[] mData[i];
        }
        delete[] mData;
        // assign new dimensions
        mNumRows = other.mNumRows;
        mNumCols = other.mNumCols;
        // allocate fresh
        mData = new double*[mNumRows];
        for (std::size_t i = 0; i < mNumRows; ++i) {
            mData[i] = new double[mNumCols];
        }
    }
    // copy data
    for (std::size_t i = 0; i < mNumRows; ++i) {
        std::copy(other.mData[i], other.mData[i] + mNumCols, mData[i]);
    }
    return *this;
}

// Destructor: free all allocated memory
Matrix::~Matrix() {
    for (std::size_t i = 0; i < mNumRows; ++i) {
        delete[] mData[i];
    }
    delete[] mData;
}

// Internal helper: swap two rows by swapping each column entry
void Matrix::SwapRows_internal(std::size_t r1, std::size_t r2) {
    assert(r1 < mNumRows && r2 < mNumRows);
    for (std::size_t j = 0; j < mNumCols; ++j) {
        std::swap(mData[r1][j], mData[r2][j]);
    }
}

// ---------------------- Accessors ----------------------

// one‐based indexing
double& Matrix::operator()(std::size_t i, std::size_t j) {
    assert(i >= 1 && i <= mNumRows);
    assert(j >= 1 && j <= mNumCols);
    return mData[i - 1][j - 1];
}

const double& Matrix::operator()(std::size_t i, std::size_t j) const {
    assert(i >= 1 && i <= mNumRows);
    assert(j >= 1 && j <= mNumCols);
    return mData[i - 1][j - 1];
}

// ---------------------- Overloaded Operators ----------------------

// unary minus
Matrix Matrix::operator-() const {
    Matrix result(mNumRows, mNumCols);
    for (std::size_t i = 0; i < mNumRows; ++i) {
        for (std::size_t j = 0; j < mNumCols; ++j) {
            result.mData[i][j] = -mData[i][j];
        }
    }
    return result;
}

// matrix + matrix
Matrix Matrix::operator+(const Matrix& rhs) const {
    assert(mNumRows == rhs.mNumRows && mNumCols == rhs.mNumCols);
    Matrix result(mNumRows, mNumCols);
    for (std::size_t i = 0; i < mNumRows; ++i) {
        for (std::size_t j = 0; j < mNumCols; ++j) {
            result.mData[i][j] = mData[i][j] + rhs.mData[i][j];
        }
    }
    return result;
}

// matrix - matrix
Matrix Matrix::operator-(const Matrix& rhs) const {
    assert(mNumRows == rhs.mNumRows && mNumCols == rhs.mNumCols);
    Matrix result(mNumRows, mNumCols);
    for (std::size_t i = 0; i < mNumRows; ++i) {
        for (std::size_t j = 0; j < mNumCols; ++j) {
            result.mData[i][j] = mData[i][j] - rhs.mData[i][j];
        }
    }
    return result;
}

// matrix * scalar
Matrix Matrix::operator*(double scalar) const {
    Matrix result(mNumRows, mNumCols);
    for (std::size_t i = 0; i < mNumRows; ++i) {
        for (std::size_t j = 0; j < mNumCols; ++j) {
            result.mData[i][j] = mData[i][j] * scalar;
        }
    }
    return result;
}

Matrix operator*(double scalar, const Matrix& M) {
    return M * scalar;
}

// matrix * matrix
Matrix Matrix::operator*(const Matrix& B) const {
    assert(mNumCols == B.mNumRows);
    Matrix result(mNumRows, B.mNumCols);
    for (std::size_t i = 0; i < mNumRows; ++i) {
        for (std::size_t j = 0; j < B.mNumCols; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < mNumCols; ++k) {
                sum += mData[i][k] * B.mData[k][j];
            }
            result.mData[i][j] = sum;
        }
    }
    return result;
}

// matrix * vector
Vector Matrix::operator*(const Vector& v) const {
    assert(mNumCols == v.Size());
    Vector result(mNumRows);
    for (std::size_t i = 0; i < mNumRows; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < mNumCols; ++j) {
            sum += mData[i][j] * v[j];
        }
        result[i] = sum;
    }
    return result;
}

// ---------------------- Other Methods ----------------------

// Transpose
Matrix Matrix::Transpose() const {
    Matrix result(mNumCols, mNumRows);
    for (std::size_t i = 0; i < mNumRows; ++i) {
        for (std::size_t j = 0; j < mNumCols; ++j) {
            result.mData[j][i] = mData[i][j];
        }
    }
    return result;
}

// Determinant (Gaussian elimination on a local copy)
// Only valid for square matrices.
double Matrix::Determinant() const {
    assert(mNumRows == mNumCols);
    std::size_t n = mNumRows;

    // Copy into a local Matrix so as not to alter *this
    Matrix A(*this);
    double det = 1.0;

    for (std::size_t i = 1; i <= n; ++i) {
        // 1) Find pivot in column i
        std::size_t pivotRow = i;
        double maxVal = std::abs(A(i, i));
        for (std::size_t r = i + 1; r <= n; ++r) {
            double val = std::abs(A(r, i));
            if (val > maxVal) {
                maxVal = val;
                pivotRow = r;
            }
        }
        // If pivotRow != i, swap those rows
        if (pivotRow != i) {
            A.SwapRows_internal(i - 1, pivotRow - 1);
            det = -det;
        }
        double pivotVal = A(i, i);
        if (std::abs(pivotVal) < 1e-12) {
            return 0.0; // singular
        }
        det *= pivotVal;

        // Eliminate below
        for (std::size_t r = i + 1; r <= n; ++r) {
            double factor = A(r, i) / pivotVal;
            for (std::size_t c = i; c <= n; ++c) {
                A(r, c) -= factor * A(i, c);
            }
        }
    }
    return det;
}

// Inverse (augmented‐matrix method; only for square, invertible)
Matrix Matrix::Inverse() const {
    assert(mNumRows == mNumCols);
    std::size_t n = mNumRows;

    // Build augmented [A | I]
    Matrix aug(n, 2 * n);
    for (std::size_t i = 1; i <= n; ++i) {
        for (std::size_t j = 1; j <= n; ++j) {
            aug(i, j) = (*this)(i, j);
        }
        for (std::size_t j = n + 1; j <= 2 * n; ++j) {
            aug(i, j) = ((j - n) == i ? 1.0 : 0.0);
        }
    }

    // Gaussian elimination → [I | A⁻¹]
    for (std::size_t i = 1; i <= n; ++i) {
        // Pivot selection
        std::size_t pivotRow = i;
        double maxVal = std::abs(aug(i, i));
        for (std::size_t r = i + 1; r <= n; ++r) {
            double val = std::abs(aug(r, i));
            if (val > maxVal) {
                maxVal = val;
                pivotRow = r;
            }
        }
        assert(maxVal > 1e-12); // must be invertible
        if (pivotRow != i) {
            aug.SwapRows_internal(i - 1, pivotRow - 1);
        }
        // Scale row i so (i,i) → 1
        double pivotVal = aug(i, i);
        for (std::size_t c = 1; c <= 2 * n; ++c) {
            aug(i, c) /= pivotVal;
        }
        // Eliminate other rows
        for (std::size_t r = 1; r <= n; ++r) {
            if (r == i) continue;
            double factor = aug(r, i);
            for (std::size_t c = 1; c <= 2 * n; ++c) {
                aug(r, c) -= factor * aug(i, c);
            }
        }
    }

    // Extract right half as inverse
    Matrix inv(n, n);
    for (std::size_t i = 1; i <= n; ++i) {
        for (std::size_t j = 1; j <= n; ++j) {
            inv(i, j) = aug(i, j + n);
        }
    }
    return inv;
}

// Pseudo-inverse (Moore-Penrose):
// If mNumRows ≥ mNumCols: A⁺ = (Aᵀ A)⁻¹ Aᵀ
// Else:                   A⁺ = Aᵀ (A Aᵀ)⁻¹
// =====================================================================
// Replace the existing Matrix::PseudoInverse() in Matrix.cpp with this:
// =====================================================================

Matrix Matrix::PseudoInverse() const {
    // Tikhonov‐regularized pseudoinverse with small lambda
    const double lambda = 1e-8;

    if (mNumRows >= mNumCols) {
        // (m×n) → compute (A^T A + λ I)^{-1} A^T
        Matrix At = this->Transpose();         // size n×m
        Matrix AtA = At * (*this);            // size n×n

        // Add λ to the diagonal of AtA (regularization)
        for (std::size_t i = 1; i <= mNumCols; ++i) {
            AtA(i, i) += lambda;
        }

        // Invert the regularized matrix
        Matrix AtA_inv = AtA.Inverse();       // now safe: (AtA + λI) is PD
        return AtA_inv * At;                  // result is n×m
    }
    else {
        // (m×n) with m < n → compute A^T (A A^T + λ I)^{-1}
        Matrix At = this->Transpose();         // size n×m
        Matrix AAt = (*this) * At;            // size m×m

        // Add λ to the diagonal of AAt
        for (std::size_t i = 1; i <= mNumRows; ++i) {
            AAt(i, i) += lambda;
        }

        // Invert the regularized matrix
        Matrix AAt_inv = AAt.Inverse();       // now safe: (AAt + λI) is PD
        return At * AAt_inv;                  // result is n×m
    }
}

// Print (for debugging)
void Matrix::Print() const {
    std::cout << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < mNumRows; ++i) {
        std::cout << "[ ";
        for (std::size_t j = 0; j < mNumCols; ++j) {
            std::cout << mData[i][j];
            if (j + 1 < mNumCols) std::cout << ", ";
        }
        std::cout << " ]\n";
    }
}
