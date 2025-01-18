//
//  Matrix.h
//  NeuralNetwork
//
//  Created by Richard Dalley on 2025-01-09.
//
#ifndef MATRIX_H
#define MATRIX_H

#include <random> // For random number generation
#include <stdexcept>
#include <iostream>
namespace NeuralNetwork{
template <typename T>
class Matrix{
private:
    std::vector<std::vector<T>> data;
    size_t rows, cols;
public:
    // Constructor and other methods...
    
    // Iterator methods
    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
    
    auto begin() const { return data.begin(); }
    auto end() const { return data.end(); }
    
    // Constructor to initialize the matrix with given dimensions
    Matrix(size_t rows, size_t cols, T defaultValue = T()) : rows(rows), cols(cols) {
        data.resize(rows, std::vector<T>(cols, defaultValue));
    }
    
    Matrix(const std::vector<T>& vec, bool asColumn = true) {
        if (asColumn) {
            rows = vec.size();
            cols = 1;
            data.resize(rows, std::vector<T>(cols));
            for (size_t i = 0; i < rows; ++i) {
                data[i][0] = vec[i];
            }
        } else {
            rows = 1;
            cols = vec.size();
            data.resize(rows, std::vector<T>(cols));
            for (size_t j = 0; j < cols; ++j) {
                data[0][j] = vec[j];
            }
        }
    }
    // Overload [] operator for row access (non-const)
    std::vector<T>& operator[](size_t row) {
        if (row >= rows) {
            throw std::out_of_range("Row index out of bounds");
        }
        return data[row];
    }

    // Overload [] operator for row access (const)
    const std::vector<T>& operator[](size_t row) const {
        if (row >= rows) {
            throw std::out_of_range("Row index out of bounds");
        }
        return data[row];
    }
      
    // Get the number of rows
    size_t getRows() const {
        return rows;
    }
    
    // Get the number of columns
    size_t getCols() const {
        return cols;
    }
    
    void fromVector(const std::vector<T>& vec, bool asColumn = true) {
        if (asColumn) {
            rows = vec.size();
            cols = 1;
            data.resize(rows, std::vector<T>(cols));
            for (size_t i = 0; i < rows; ++i) {
                data[i][0] = vec[i];
            }
        } else {
            rows = 1;
            cols = vec.size();
            data.resize(rows, std::vector<T>(cols));
            for (size_t j = 0; j < cols; ++j) {
                data[0][j] = vec[j];
            }
        }
    }

    std::vector<T> toVector() const {
        std::vector<T> result;

        if (rows == 1) {
            // Row vector
            result = data[0];
        } else if (cols == 1) {
            // Column vector
            result.reserve(rows);
            for (size_t i = 0; i < rows; ++i) {
                result.push_back(data[i][0]);
            }
        } else {
            throw std::invalid_argument("Matrix is not a vector (1 row or 1 column).");
        }

        return result;
    }

    Matrix<T> transpose() const {
        Matrix<T> transposed(cols, rows);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                transposed[j][i] = this->data[i][j];
            }
        }
        return transposed;
    }
    
    Matrix<T> operator*(const Matrix<T>& other) const {
        // Case 1: Element-wise multiplication
        if (rows == other.getRows() && cols == other.getCols()) {
            Matrix<T> result(rows, cols);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result[i][j] = this->data[i][j] * other[i][j];
                }
            }
            return result;
        }
        
        // Case 2: Traditional matrix multiplication
        if (cols == other.getRows()) {
            Matrix<T> result(rows, other.getCols());
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < other.getCols(); ++j) {
                    T sum = T(); // Default initialize the sum
                    for (size_t k = 0; k < cols; ++k) {
                        sum += this->data[i][ k] * other[k][j];
                    }
                    result[i][j] = sum;
                }
            }
            return result;
        }
        
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }
    
    Matrix<T> operator*(T scalar) const {
        Matrix<T> result(rows, cols);
        
        // Scale each element by the scalar
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] = this->data[i][j] * scalar;
            }
        }
        
        return result; // Return the new scaled matrix
    }
    
    Matrix<T>& operator*=(T scalar) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                this->data[i][j] *= scalar;
            }
        }
        return *this; // Return reference to the modified matrix
    }
    
    Matrix<T> outer(const Matrix<T>& other) const {
        // Ensure this is a column vector
        if (this->cols != 1) {
            throw std::invalid_argument("First matrix must be a column vector for outer product.");
        }
        
        // Ensure the other matrix is a row vector
        if (other.rows != 1) {
            throw std::invalid_argument("Second matrix must be a row vector for outer product.");
        }
        
        // Create result matrix with dimensions (rows of this x cols of other)
        Matrix<T> result(this->rows, other.cols);
        
        // Perform outer product
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                result[i][j] = this->data[i][ 0] * other[0][ j];
            }
        }
        
        return result;
    }
    
    Matrix<T> operator-(const Matrix<T>& other) const {
        // Ensure the dimensions match for subtraction
        if (rows != other.getRows() || cols != other.getCols()) {
            throw std::invalid_argument("Matrix dimensions do not match for subtraction");
        }
        
        // Create a result matrix with the same dimensions
        Matrix<T> result(rows, cols);
        
        // Perform element-wise subtraction
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] = this->data[i][j] - other[i][j];
            }
        }
        
        return result;
    }
    
    Matrix<T>& operator+=(const Matrix<T>& other) {
        // Ensure the dimensions match for addition
        if (rows != other.getRows() || cols != other.getCols()) {
            throw std::invalid_argument("Matrix dimensions do not match for addition");
        }
        
        // Perform element-wise addition
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                this->data[i][j] += other[i][j];
            }
        }
        
        return *this; // Return reference to the modified matrix
    }
    
    // Print the matrix
    void print() const {
        for (const auto& row : data) {
            for (const auto& val : row) {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }
    }
};

// MatrixRandomizer class to add randomization functionality
template <typename T>
class MatrixRandomizer {
private:
    Matrix<T>& matrix;
    
public:
    // Constructor accepts a reference to the Matrix object
    MatrixRandomizer(Matrix<T>& matrix) : matrix(matrix) {}
    
    // Method to fill the matrix with random values in the range [-1.0, 1.0]
    void fillRandom() {
        std::random_device rd;  // Seed
        std::mt19937 gen(rd()); // Mersenne Twister RNG
        std::normal_distribution<> dis(0.0, 1.0); // Random floats in [-1.0, 1.0]
        
        for (size_t i = 0; i < matrix.getRows(); ++i) {
            for (size_t j = 0; j < matrix.getCols(); ++j) {
                matrix[i][ j] = dis(gen); // Assign random value
            }
        }
    }
};


}

#endif //MATRIX_H