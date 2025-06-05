#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "Vector.hpp"
#include <cstddef>
#include <cassert>
#include <iostream>

class Matrix {
private:
    std::size_t mNumRows;
    std::size_t mNumCols;
    double**    mData;   // mData[i] is a double* to the i-th row

    // (internal) swap two rows by swapping each column entry
    void SwapRows_internal(std::size_t r1, std::size_t r2);

public:
    // --- Constructors / Destructor ---
    Matrix(std::size_t rows, std::size_t cols);
    Matrix(const Matrix& other);            // copy constructor
    Matrix& operator=(const Matrix& other); // copy assignment
    ~Matrix();

    // --- Accessors ---
    std::size_t Rows() const { return mNumRows; }
    std::size_t Cols() const { return mNumCols; }

    // one‐based indexing: A(i,j) ↦ mData[i−1][j−1]
    double&       operator()(std::size_t i, std::size_t j);
    const double& operator()(std::size_t i, std::size_t j) const;

    // --- Overloaded operators ---
    // unary minus
    Matrix operator-() const;

    // matrix + matrix
    Matrix operator+(const Matrix& rhs) const;

    // matrix - matrix
    Matrix operator-(const Matrix& rhs) const;

    // matrix * scalar
    Matrix operator*(double scalar) const;
    friend Matrix operator*(double scalar, const Matrix& M);

    // matrix * matrix
    Matrix operator*(const Matrix& B) const;

    // matrix * vector
    Vector operator*(const Vector& v) const;

    // --- Other methods ---
    Matrix Transpose() const;
    double Determinant() const;      // only for square matrices
    Matrix Inverse() const;          // only for square, invertible matrices
    Matrix PseudoInverse() const;    // Moore-Penrose pseudoinverse

    // --- Utility ---
    void Print() const;  // prints the matrix entries (for debugging)
};

#endif // MATRIX_HPP
