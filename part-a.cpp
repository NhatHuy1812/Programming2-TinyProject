#include "Vector.hpp"
#include "Matrix.hpp"
#include "LinearSystem.hpp"
#include "PosSymLinSystem.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <cctype>
#include <vector>

// ------------------------------------------------------------------
// Helper: Read a square or rectangular linear system (A and b) from a
// single text file, skipping blank lines or lines beginning with '#'.  
// File format (after skipping comments/blanks):
//
//   <m> <n>          (two integers: # rows of A, # cols of A)
//   a11 a12 ... a1n  (first row of A)
//   a21 a22 ... a2n
//   ...
//   a_m1 a_m2 ... a_mn
//   b1
//   b2
//   ...
//   b_m
//
// - Lines beginning with '#' (after optional leading whitespace) are ignored.
// - Blank lines (or lines with only whitespace) are ignored.
//
// On success, returns pair { A_matrix, b_vector }.
// Throws std::runtime_error on any I/O or format error.
// ------------------------------------------------------------------
static void ReadLinearSystemFromFile(const std::string& filename,
                                     Matrix& outA,
                                     Vector& outb)
{
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Could not open input file: " + filename);
    }

    auto trim_leading = [&](const std::string& s) {
        std::size_t i = 0;
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) {
            ++i;
        }
        return s.substr(i);
    };

    std::string line;
    std::size_t m = 0, n = 0;
    bool dimsRead = false;

    // 1) Find first non-blank/non-comment line, parse m and n
    while (std::getline(infile, line)) {
        std::string t = trim_leading(line);
        if (t.empty() || t[0] == '#') {
            continue; // skip blank or comment
        }
        std::istringstream iss(t);
        if (!(iss >> m >> n)) {
            throw std::runtime_error(
                "Failed to parse matrix dimensions (m n) from line: \"" + line + "\""
            );
        }
        dimsRead = true;
        break;
    }
    if (!dimsRead) {
        throw std::runtime_error("No dimensions found in file: " + filename);
    }
    if (m == 0 || n == 0) {
        throw std::runtime_error("Matrix dimensions must be positive in file: " + filename);
    }

    // 2) Read the next m rows, each with n entries, for matrix A
    Matrix A(m, n);
    std::size_t rowsRead = 0;
    while (rowsRead < m && std::getline(infile, line)) {
        std::string t = trim_leading(line);
        if (t.empty() || t[0] == '#') {
            continue; // skip blanks or comments
        }
        std::istringstream iss(t);
        for (std::size_t col = 1; col <= n; ++col) {
            double val;
            if (!(iss >> val)) {
                std::ostringstream oss;
                oss << "Failed to read A(" << (rowsRead+1) << "," << col
                    << ") from line: \"" << line << "\"";
                throw std::runtime_error(oss.str());
            }
            A(rowsRead + 1, col) = val;
        }
        // Optional: if (iss >> leftover) { warn about extra tokens; }
        ++rowsRead;
    }
    if (rowsRead < m) {
        throw std::runtime_error(
            "Expected " + std::to_string(m) + " rows for A, but only read "
            + std::to_string(rowsRead) + " in file: " + filename
        );
    }

    // 3) Read the next m entries (one per line) for vector b
    Vector b(m);
    std::size_t entriesRead = 0;
    while (entriesRead < m && std::getline(infile, line)) {
        std::string t = trim_leading(line);
        if (t.empty() || t[0] == '#') {
            continue; // skip blanks or comments
        }
        std::istringstream iss(t);
        double val;
        if (!(iss >> val)) {
            std::ostringstream oss;
            oss << "Failed to read b(" << (entriesRead + 1)
                << ") from line: \"" << line << "\"";
            throw std::runtime_error(oss.str());
        }
        b(entriesRead + 1) = val;
        ++entriesRead;
    }
    if (entriesRead < m) {
        throw std::runtime_error(
            "Expected " + std::to_string(m) + " entries for b, but only read "
            + std::to_string(entriesRead) + " in file: " + filename
        );
    }

    infile.close();
    outA = std::move(A);
    outb = std::move(b);
}

// ------------------------------------------------------------------
// main: read A & b from a single file, then solve using specified mode.
// Usage:  program.exe <system_file.txt> <mode>
//    <mode> = "gauss" (Gaussian elimination  or pseudoinverse fallback)
//           = "cg"    (Conjugate Gradient, requires A SPD)
// ------------------------------------------------------------------
int main(int argc, char* argv[]) {
    try {
        if (argc != 3) {
            std::cerr << "Usage:\n"
                      << "  " << argv[0] << " <system_file.txt> <mode>\n\n"
                      << "  <mode> = 'gauss'   (Gaussian elimination or pseudoinverse)\n"
                      << "         = 'cg'      (Conjugate Gradient, requires A SPD)\n";
            return 1;
        }

        const std::string filename = argv[1];
        const std::string mode     = argv[2];

        // Read both A (m×n) and b (length m) from the same file
        Matrix A(1,1);  // placeholders, will be replaced
        Vector b(1);
        ReadLinearSystemFromFile(filename, A, b);

        std::size_t m = A.Rows();
        std::size_t n = A.Cols();
        std::size_t bs = b.Size();  // should equal m

        if (bs != m) {
            throw std::runtime_error(
                "Vector length (" + std::to_string(bs) +
                ") does not match number of rows (" + std::to_string(m) + ")."
            );
        }

        if (mode == "gauss") {
            if (m == n) {
                // Square system → Gaussian elimination
                LinearSystem sys(A, b);
                Vector x = sys.Solve();
                std::cout << "Solution via Gaussian elimination:\n";
                x.Print();
            }
            else {
                // Non‐square: use pseudoinverse A^+ b
                std::cout << "Matrix is not square. Using pseudoinverse A^+ b\n";
                Matrix A_pinv = A.PseudoInverse();  // dimension n×m
                if (A_pinv.Cols() != bs) {
                    throw std::runtime_error(
                        "Cannot multiply A^+ (" + std::to_string(A_pinv.Rows()) + "×"
                        + std::to_string(A_pinv.Cols()) + ") by b (" 
                        + std::to_string(bs) + ")"
                    );
                }
                Vector x = A_pinv * b;  // size n
                std::cout << "Solution via pseudoinverse:\n";
                x.Print();
            }
        }
        else if (mode == "cg") {
            // Conjugate Gradient requires square (m==n) and SPD
            if (m != n) {
                throw std::runtime_error("CG requires square A; got " +
                    std::to_string(m) + "x" + std::to_string(n));
            }
            PosSymLinSystem psys(A, b);
            Vector x = psys.Solve();
            std::cout << "Solution via Conjugate Gradient (SPD):\n";
            x.Print();
        }
        else {
            throw std::runtime_error("Unknown mode: '" + mode + "'. Use 'gauss' or 'cg'.");
        }
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
