// ---------------------------
// File: Regression.cpp
// ---------------------------

#include "Vector.hpp"
#include "Matrix.hpp"
#include "LinearSystem.hpp"  // only for Vector/Matrix types

#include <array>        // for std::array<double,6>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cmath>

// --------------------------------------------------------------------------------
// Helper: Read exactly 209 lines from "machine.data", skipping blank/comment lines.
// Each non‐blank, non‐comment line has 10 comma‐separated fields:
//   [0] vendor_name    (ignore)
//   [1] model_name     (ignore)
//   [2] MYCT   (int)   → feature #1
//   [3] MMIN   (int)   → feature #2
//   [4] MMAX   (int)   → feature #3
//   [5] CACH   (int)   → feature #4
//   [6] CHMIN  (int)   → feature #5
//   [7] CHMAX  (int)   → feature #6
//   [8] PRP    (int)   → target y
//   [9] ERP    (ignore)
//
// We skip any line that is empty or begins with '#' (after trimming leading spaces).
// Expect exactly 209 data lines. On success, dataX[i] is the six features, dataY[i] is PRP.
//
// Throws std::runtime_error if something goes wrong (I/O or parse).
// --------------------------------------------------------------------------------
static void LoadCPUDataset(
    const std::string& filename,
    std::vector<std::array<double,6>>& dataX,
    std::vector<double>& dataY
) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Could not open dataset file: " + filename);
    }

    auto trim_leading = [&](const std::string& s) {
        std::size_t i = 0;
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) {
            ++i;
        }
        return s.substr(i);
    };

    std::string line;
    dataX.clear();
    dataY.clear();

    while (std::getline(infile, line)) {
        std::string t = trim_leading(line);
        if (t.empty() || t[0] == '#') {
            continue; // skip blank or comment
        }

        // Split on commas
        std::vector<std::string> tokens;
        {
            std::istringstream iss(t);
            std::string tok;
            while (std::getline(iss, tok, ',')) {
                tokens.push_back(tok);
            }
        }
        if (tokens.size() != 10) {
            std::ostringstream oss;
            oss << "Expected 10 fields but found " << tokens.size()
                << " in line: \"" << t << "\"";
            throw std::runtime_error(oss.str());
        }

        // Parse the six integer features into doubles
        std::array<double,6> features;
        for (int i = 0; i < 6; ++i) {
            int val;
            try {
                val = std::stoi(tokens[2 + i]);
            }
            catch (...) {
                std::ostringstream oss;
                oss << "Failed to parse integer feature #" << (i+1)
                    << " from token: \"" << tokens[2 + i]
                    << "\" in line: \"" << t << "\"";
                throw std::runtime_error(oss.str());
            }
            features[i] = static_cast<double>(val);
        }
        dataX.push_back(features);

        // Parse the target PRP as double
        int prp;
        try {
            prp = std::stoi(tokens[8]);
        }
        catch (...) {
            std::ostringstream oss;
            oss << "Failed to parse PRP from token: \"" << tokens[8]
                << "\" in line: \"" << t << "\"";
            throw std::runtime_error(oss.str());
        }
        dataY.push_back(static_cast<double>(prp));
    }

    infile.close();

    if (dataX.size() != 209 || dataY.size() != 209) {
        std::ostringstream oss;
        oss << "Expected to read exactly 209 data lines, but found " << dataX.size();
        throw std::runtime_error(oss.str());
    }
}

// --------------------------------------------------------------------------------
// Compute the column‐wise mean and standard deviation of X_train (N_train × 6).
// Return mean[j], stddev[j] for j=0…5.  If any stddev[j] < 1e-12, set it to 1.0.
// --------------------------------------------------------------------------------
static void ComputeFeatureStatistics(
    const Matrix& X_train,
    std::array<double,6>& mean,
    std::array<double,6>& stddev
) {
    std::size_t N = X_train.Rows();
    // 1) Column means
    for (std::size_t j = 0; j < 6; ++j) {
        double sum = 0.0;
        for (std::size_t i = 1; i <= N; ++i) {
            sum += X_train(i, j+1);
        }
        mean[j] = sum / static_cast<double>(N);
    }
    // 2) Column stddevs
    for (std::size_t j = 0; j < 6; ++j) {
        double sq_sum = 0.0;
        for (std::size_t i = 1; i <= N; ++i) {
            double diff = X_train(i, j+1) - mean[j];
            sq_sum += diff * diff;
        }
        stddev[j] = std::sqrt(sq_sum / static_cast<double>(N));
        if (stddev[j] < 1e-12) {
            stddev[j] = 1.0;
        }
    }
}

// --------------------------------------------------------------------------------
// Normalize each entry of X (m × 6) in‐place:  X(i,j) = (orig - mean[j-1]) / stddev[j-1].
// mean[] and stddev[] are length 6, zero‐indexed.  X is 1‐indexed.
// --------------------------------------------------------------------------------
static void NormalizeFeatures(
    Matrix& X,
    const std::array<double,6>& mean,
    const std::array<double,6>& stddev
) {
    std::size_t m = X.Rows();
    for (std::size_t i = 1; i <= m; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            double orig = X(i, j+1);
            X(i, j+1) = (orig - mean[j]) / stddev[j];
        }
    }
}

// --------------------------------------------------------------------------------
// Compute RMSE = sqrt( (1/m) * Σ ( (φᵀ x_i) - y_i )² ) for X (m×6), φ (6), y (m).
// X is 1‐indexed, φ and y are 1‐indexed.
// --------------------------------------------------------------------------------
static double ComputeRMSE(const Matrix& X, const Vector& phi, const Vector& y) {
    std::size_t m = X.Rows();
    double sumSq = 0.0;
    for (std::size_t i = 1; i <= m; ++i) {
        double y_pred = 0.0;
        for (std::size_t j = 1; j <= 6; ++j) {
            y_pred += phi(j) * X(i, j);
        }
        double err = y_pred - y(i);
        sumSq += err * err;
    }
    return std::sqrt(sumSq / static_cast<double>(m));
}

// --------------------------------------------------------------------------------
// main: read "machine.data", split 80/20, run 10 epochs of batch gradient descent
// on normalized features, print training RMSE each epoch, then final test RMSE.
//
// Usage:
//   ./cpu_regression machine.data
// --------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    try {
        if (argc != 2) {
            std::cerr << "Usage:\n"
                      << "  " << argv[0] << " <machine.data>\n";
            return 1;
        }
        const std::string filename = argv[1];

        // 1) Load all 209 instances into allX (209×6) and allY (209)
        std::vector<std::array<double,6>> allX;
        std::vector<double>               allY;
        LoadCPUDataset(filename, allX, allY);

        // 2) Build indices [0..208], shuffle, split 80%/20% (seed 12345)
        std::vector<int> idx(209);
        for (int i = 0; i < 209; ++i) idx[i] = i;
        std::mt19937 rng(12345);
        std::shuffle(idx.begin(), idx.end(), rng);

        const int N_train = static_cast<int>(std::floor(0.8 * 209)); // 167
        const int N_test  = 209 - N_train;                           // 42

        // 3) Build X_train (167×6) and y_train (167)
        Matrix X_train(N_train, 6);
        Vector y_train(N_train);
        for (int i = 0; i < N_train; ++i) {
            int orig = idx[i];
            for (int j = 0; j < 6; ++j) {
                X_train(i + 1, j + 1) = allX[orig][j];
            }
            y_train(i + 1) = allY[orig];
        }

        // 4) Build X_test (42×6) and y_test (42)
        Matrix X_test(N_test, 6);
        Vector y_test(N_test);
        for (int i = 0; i < N_test; ++i) {
            int orig = idx[N_train + i];
            for (int j = 0; j < 6; ++j) {
                X_test(i + 1, j + 1) = allX[orig][j];
            }
            y_test(i + 1) = allY[orig];
        }

        // 5) Compute column stats (mean, stddev) on X_train
        std::array<double,6> mean, stddev;
        ComputeFeatureStatistics(X_train, mean, stddev);

        // 6) Normalize both X_train and X_test
        NormalizeFeatures(X_train, mean, stddev);
        NormalizeFeatures(X_test,  mean, stddev);

        // 7) Initialize φ = zero vector (length 6) and choose learning rate = 0.01
        Vector phi(6);
        for (std::size_t j = 1; j <= 6; ++j) {
            phi(j) = 0.0;
        }
        const double lr = 0.2;   // <<<<< much larger than before

        // 8) Run 10 epochs of batch gradient descent
        for (int epoch = 1; epoch <= 35; ++epoch) {
            // a) Accumulate gradient (length 6) = Σ_i [ 2 (φᵀ x_i - y_i) · x_i ]
            std::array<double,6> grad{};
            for (std::size_t j = 0; j < 6; ++j) {
                grad[j] = 0.0;
            }

            for (int i = 1; i <= N_train; ++i) {
                // Compute y_pred_i = φᵀ x_i
                double y_pred = 0.0;
                for (int j = 1; j <= 6; ++j) {
                    y_pred += phi(j) * X_train(i, j);
                }
                double resid = y_pred - y_train(i);
                // Add 2 * resid * x_i(j) to gradient[j-1]
                for (int j = 1; j <= 6; ++j) {
                    grad[j-1] += 2.0 * resid * X_train(i, j);
                }
            }

            // b) Scale gradient by (1 / N_train) and update φ
            for (int j = 1; j <= 6; ++j) {
                double g = grad[j-1] / static_cast<double>(N_train);
                phi(j) -= lr * g;
            }

            // c) Compute and print training RMSE
            double train_rmse = ComputeRMSE(X_train, phi, y_train);
            std::cout << "Epoch " << epoch << " - Training RMSE = " 
                      << train_rmse << "\n";
        }
        std::cout << "\n";

        // 9) After 10 epochs, compute and print test RMSE
        double test_rmse = ComputeRMSE(X_test, phi, y_test);
        std::cout << "Final Test set RMSE = " << test_rmse
                  << " (over " << N_test << " examples)\n\n";

        // 10) Print final φ values
        std::cout << "Learned parameters (phi) after 10 epochs:\n";
        for (int j = 1; j <= 6; ++j) {
            std::cout << "  phi[" << j << "] = " << phi(j) << "\n";
        }
        std::cout << "\n";
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
