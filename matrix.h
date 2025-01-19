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
    std::vector<T> data;
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
        data.resize(rows * cols, defaultValue);
    }

    Matrix(const std::vector<T>& vec, bool asColumn = true) {
    if (asColumn) {
        // Treat the input vector as a column vector
        rows = vec.size();
        cols = 1;
    } else {
        // Treat the input vector as a row vector
        rows = 1;
        cols = vec.size();
    }

    // Resize the flat vector and copy data
    data.resize(rows * cols);
    for (size_t i = 0; i < vec.size(); ++i) {
        data[i] = vec[i];
    }
}

    // Method to fill the matrix with random values in the range [-1.0, 1.0]
    void fillRandom() {
        std::random_device rd;  // Seed
        std::mt19937 gen(rd()); // Mersenne Twister RNG
        std::normal_distribution<> dis(0.0, 1.0); // Random floats in [-1.0, 1.0]

        for (size_t i = 0; i < rows * cols; ++i) {
            data[i] = dis(gen); // Assign random value
        }
    }

    //overload () to make it more intuitive, and to work properly with a flat-matrix
    T& operator()(size_t row, size_t col) {
        if (row >= rows || col >= cols) {
            throw std::out_of_range("Index out of bounds");
        }
        return data[row * cols + col]; // Calculate the flat index
    }

    //const implementation of overload ()
    const T& operator()(size_t row, size_t col) const {
        if (row >= rows || col >= cols) {
            throw std::out_of_range("Index out of bounds");
        }
        return data[row * cols + col]; // Calculate the flat index
    }

    // Get the number of rows
    size_t getRows() const {
        return rows;
    }
    
    // Get the number of columns
    size_t getCols() const {
        return cols;
    }
    
    // Copy the values from a vector into the matrix. 
    // By default, the vector is treated as a column vector (1 element per row).
    // If `asColumn` is false, the vector is treated as a row vector (1 row, multiple columns).
    void fromVector(const std::vector<T>& vec, bool asColumn = true) {
        if (asColumn) {
            // Treat the input vector as a column vector
            rows = vec.size();
            cols = 1;
            data.resize(rows * cols);  // Resize the flat vector
            for (size_t i = 0; i < rows; ++i) {
                data[i] = vec[i];  // Fill column-wise
            }
        } else {
            // Treat the input vector as a row vector
            rows = 1;
            cols = vec.size();
            data.resize(rows * cols);  // Resize the flat vector
            for (size_t j = 0; j < cols; ++j) {
                data[j] = vec[j];  // Fill row-wise
            }
        }
    }

    std::vector<T> extract() const {
        std::vector<T> result;

        if (rows == 1) {
            // Row vector: copy all elements in order
            result = data;  // The entire flat data is already the row
        } else if (cols == 1) {
            // Column vector: extract elements column-wise
            result.reserve(rows);  // Preallocate space for efficiency
            for (size_t i = 0; i < rows; ++i) {
                result.push_back(data[i]);  // Column data is sequential in flat storage
            }
        } else {
            throw std::invalid_argument("Matrix is not a vector (1 row or 1 column).");
        }

        return result;
    }

    Matrix<T> transpose() const {
        // Create a new transposed matrix with swapped rows and cols
        Matrix<T> transposed(cols, rows);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                // Transpose: element at (i, j) becomes (j, i)
                transposed(j, i) = this->data[i * cols + j];
            }
        }

        return transposed;
    }   

    Matrix<T> operator*(const Matrix<T>& other) const {
        if (rows != other.getRows() || cols != other.getCols()) {
            throw std::invalid_argument("Operator *:Matrix dimensions must match for element-wise multiplication");
        }

        Matrix<T> result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = this->data[i * cols + j] * other.data[i * cols + j];  // Element-wise multiplication
            }
        }
        return result;
    }

    Matrix<T> dot(const Matrix<T>& other) const {
        if (cols != other.getRows()) {
            throw std::invalid_argument("matMul: Matrix dimensions do not match for multiplication");
        }

        Matrix<T> result(rows, other.getCols());
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.getCols(); ++j) {
                T sum = T();  // Initialize sum to zero
                for (size_t k = 0; k < cols; ++k) {
                    size_t lhsIndex = i * cols + k;  // Flat index for left-hand side
                    size_t rhsIndex = k * other.getCols() + j;  // Flat index for right-hand side
                    sum += this->data[lhsIndex] * other.data[rhsIndex];
                }
                result(i, j) = sum;  // Store result
            }
        }
        return result;
    }

    Matrix<T> operator*(T scalar) const {
        Matrix<T> result(rows, cols);

        // Scale each element by the scalar
        for (size_t i = 0; i < rows * cols; ++i) {
            result.data[i] = this->data[i] * scalar;
        }

        return result;  // Return the new scaled matrix
    }    
 
    Matrix<T>& operator*=(T scalar) {
        // Scale each element by the scalar
        for (size_t i = 0; i < rows * cols; ++i) {
            this->data[i] *= scalar;
        }

        return *this;  // Return reference to the modified matrix
    }    
 
    // Creates an outer product of two matrices, typically used for operations 
    // such as updating weights in neural networks or constructing higher-dimensional tensors.
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
                size_t indexResult = i * other.cols + j;  // Index in the result matrix
                size_t indexA = i;  // Since this is a column vector, row index matches flat index
                size_t indexB = j;  // Since the other is a row vector, col index matches flat index

                result.data[indexResult] = this->data[indexA] * other.data[indexB];
            }
        }

        return result;
    }    

    // subtract one matrix from another
    Matrix<T> operator-(const Matrix<T>& other) const {
        // Ensure the dimensions match for subtraction
        if (rows != other.getRows() || cols != other.getCols()) {
            throw std::invalid_argument("Matrix dimensions do not match for subtraction");
        }

        // Create a result matrix with the same dimensions
        Matrix<T> result(rows, cols);

        // Perform element-wise subtraction
        size_t totalSize = rows * cols;  // Calculate total number of elements
        for (size_t i = 0; i < totalSize; ++i) {
            result.data[i] = this->data[i] - other.data[i];
        }

        return result;
    }    

    // append another matrix to this
    Matrix<T>& operator+=(const Matrix<T>& other) {
        // Ensure the dimensions match for addition
        if (rows != other.getRows() || cols != other.getCols()) {
            throw std::invalid_argument("Matrix dimensions do not match for addition");
        }

        // Perform element-wise addition
        size_t totalSize = rows * cols;  // Total number of elements
        for (size_t i = 0; i < totalSize; ++i) {
            this->data[i] += other.data[i];
        }

        return *this; // Return reference to the modified matrix
    }

    // return the sum of two operators
    Matrix<T> operator+(const Matrix<T>& other) const {
        // Ensure the dimensions match for addition
        if (rows != other.getRows() || cols != other.getCols()) {
            throw std::invalid_argument("Matrix dimensions do not match for addition");
        }

        // Create a new matrix to store the result
        Matrix<T> result(rows, cols);

        // Perform element-wise addition
        size_t totalSize = rows * cols;
        for (size_t i = 0; i < totalSize; ++i) {
            result.data[i] = this->data[i] + other.data[i];
        }

        return result; // Return the new matrix
    }

    // Print the matrix
    void print() const {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                std::cout << data[i * cols + j] << " ";
            }
            std::cout << "\n";
        }
    }
};


}

#endif //MATRIX_H