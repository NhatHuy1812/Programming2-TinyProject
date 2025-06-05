#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <cstddef>
#include <cassert>
#include <iostream>

class Vector {
private:
    std::size_t mSize;
    double*     mData;

public:
    // --- Constructors / Destructor ---
    explicit Vector(std::size_t n);
    Vector(const Vector& other);            // copy constructor
    Vector& operator=(const Vector& other); // copy assignment
    ~Vector();

    // --- Accessors ---
    std::size_t Size() const { return mSize; }

    // zero‐based indexing (with bounds check)
    double&       operator[](std::size_t idx);
    const double& operator[](std::size_t idx) const;

    // one‐based indexing (with bounds check)
    double&       operator()(std::size_t i);
    const double& operator()(std::size_t i) const;

    // --- Unary & Binary operators ---
    Vector operator-() const;                         // unary minus
    Vector operator+(const Vector& rhs) const;        // vector + vector
    Vector operator-(const Vector& rhs) const;        // vector - vector
    Vector operator*(double scalar) const;            // vector * scalar
    friend Vector operator*(double scalar, const Vector& v);

    // Dot product (member function)
    double Dot(const Vector& other) const;

    // --- Utility ---
    void Print() const;  // prints entries for debugging
};

#endif // VECTOR_HPP
