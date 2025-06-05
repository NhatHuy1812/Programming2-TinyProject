#include "Vector.hpp"
#include <algorithm> // for std::copy
#include <iomanip>   // for std::setprecision

// Constructor: allocate and zero‐initialize
Vector::Vector(std::size_t n)
    : mSize(n),
      mData(nullptr)
{
    assert(n > 0);
    mData = new double[mSize];
    for (std::size_t i = 0; i < mSize; ++i) {
        mData[i] = 0.0;
    }
}

// Copy constructor (deep copy)
Vector::Vector(const Vector& other)
    : mSize(other.mSize),
      mData(nullptr)
{
    mData = new double[mSize];
    std::copy(other.mData, other.mData + mSize, mData);
}

// Copy‐assignment operator (deep copy; handle self‐assignment)
Vector& Vector::operator=(const Vector& other) {
    if (this == &other) {
        return *this;
    }
    if (mSize != other.mSize) {
        delete[] mData;
        mSize = other.mSize;
        mData = new double[mSize];
    }
    std::copy(other.mData, other.mData + mSize, mData);
    return *this;
}

// Destructor
Vector::~Vector() {
    delete[] mData;
}

// zero‐based operator[]
double& Vector::operator[](std::size_t idx) {
    assert(idx < mSize);
    return mData[idx];
}

const double& Vector::operator[](std::size_t idx) const {
    assert(idx < mSize);
    return mData[idx];
}

// one‐based operator()
double& Vector::operator()(std::size_t i) {
    assert(i >= 1 && i <= mSize);
    return mData[i - 1];
}

const double& Vector::operator()(std::size_t i) const {
    assert(i >= 1 && i <= mSize);
    return mData[i - 1];
}

// unary minus
Vector Vector::operator-() const {
    Vector result(mSize);
    for (std::size_t i = 0; i < mSize; ++i) {
        result.mData[i] = -mData[i];
    }
    return result;
}

// vector + vector
Vector Vector::operator+(const Vector& rhs) const {
    assert(mSize == rhs.mSize);
    Vector result(mSize);
    for (std::size_t i = 0; i < mSize; ++i) {
        result.mData[i] = mData[i] + rhs.mData[i];
    }
    return result;
}

// vector - vector
Vector Vector::operator-(const Vector& rhs) const {
    assert(mSize == rhs.mSize);
    Vector result(mSize);
    for (std::size_t i = 0; i < mSize; ++i) {
        result.mData[i] = mData[i] - rhs.mData[i];
    }
    return result;
}

// vector * scalar
Vector Vector::operator*(double scalar) const {
    Vector result(mSize);
    for (std::size_t i = 0; i < mSize; ++i) {
        result.mData[i] = mData[i] * scalar;
    }
    return result;
}

// scalar * vector (friend)
Vector operator*(double scalar, const Vector& v) {
    return v * scalar;
}

// dot product
double Vector::Dot(const Vector& other) const {
    assert(mSize == other.mSize);
    double sum = 0.0;
    for (std::size_t i = 0; i < mSize; ++i) {
        sum += mData[i] * other.mData[i];
    }
    return sum;
}

// Print (for debugging)
void Vector::Print() const {
    std::cout << "[ ";
    for (std::size_t i = 0; i < mSize; ++i) {
        std::cout << std::fixed << std::setprecision(6) << mData[i];
        if (i + 1 < mSize) std::cout << ", ";
    }
    std::cout << " ]\n";
}
