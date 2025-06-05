# Programming2-TinyProject

A C++ framework for vectors, matrices, and linear‐system solvers, plus two demo applications:

- **part-a**: solve \(A x = b\) for square, under- or over-determined systems  
- **part-b**: perform linear regression on the UCI CPU-performance dataset

## Components

### 1. Vector (include/linear_algebra/Vector.hpp / src/Vector.cpp)

- `Vector(size_t n)` – allocate & zero-initialize  
- Copy-constructor, assignment, destructor  
- `operator[]` (0-based), `operator()` (1-based) access  
- `+`, `-`, `*` (scalar), unary `-`, `Dot(...)`  
- `operator<<` for printing

### 2. Matrix (include/linear_algebra/Matrix.hpp / src/Matrix.cpp)

- `Matrix(size_t rows, size_t cols)` – allocate & zero-initialize  
- Copy-constructor, assignment, destructor  
- `operator()(i,j)` (1-based) access  
- `+`, `-`, `*` (scalar, matrix, vector)  
- `Transpose()`, `Determinant()`, `Inverse()`, `PseudoInverse()`  
- `operator<<` for printing

### 3. LinearSystem (include/linear_algebra/LinearSystem.hpp / src/LinearSystem.cpp)

- Constructor from `Matrix` & `Vector` (deep copy)  
- Destructor (release memory)  
- `Solve()` – Gaussian elimination w/ partial pivoting → returns solution `Vector`

### 4. PosSymLinSystem (include/linear_algebra/PosSymLinSystem.hpp / src/PosSymLinSystem.cpp)

- Derived from `LinearSystem`  
- Overrides `Solve()` using Conjugate Gradient (for SPD matrices)

---

## Demo Applications

### part-a: General \(A x = b\) solver

- Source: `part-a.cpp`  
- Input format (`part-a-input.txt`):
  ```
  m n
  A (m×n rows)
  b (m entries)
  ```
- Invocation:
  ```powershell
  g++ -std=c++11 -O2 Vector.cpp Matrix.cpp LinearSystem.cpp part-a.cpp -o part-a.exe
  .\part-a.exe part-a-input.txt gauss
  ```
  - `gauss` : Gaussian elimination (square) or pseudoinverse (non-square)  
  - `cg`    : Conjugate gradient (square, symmetric)

### part-b: CPU-Performance Regression

- Source: `part-b.cpp`  
- Dataset in `machine.data` (UCI CPU data)  
- Loads, normalizes, splits 80/20, runs batch gradient descent  
- Invocation:
  ```powershell
  g++ -std=c++11 -O2 Vector.cpp Matrix.cpp LinearSystem.cpp part-b.cpp -o part-b.exe
  .\part-b.exe machine.data
  ```

---

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
