#ifndef LINEAR_SYSTEM_HPP
#define LINEAR_SYSTEM_HPP

#include "Matrix.hpp"
#include "Vector.hpp"
#include <cstddef>
#include <cassert>

class LinearSystem {
protected:
    std::size_t mSize;  // dimension of the square system
    Matrix*     mpA;    // pointer to the coefficient matrix (size mSize×mSize)
    Vector*     mpb;    // pointer to the RHS vector (size mSize)

    // disable default and copy
    LinearSystem() = delete;
    LinearSystem(const LinearSystem&) = delete;
    LinearSystem& operator=(const LinearSystem&) = delete;

public:
    // Construct from A and b (makes deep copies)
    LinearSystem(const Matrix& A, const Vector& b);

    virtual ~LinearSystem();

    // Solve Ax = b via Gaussian elimination with partial pivoting
    // Returns a Vector of size mSize containing x
    virtual Vector Solve() const;
};

#endif // LINEAR_SYSTEM_HPP
