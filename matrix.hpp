// https://github.com/douglasrizzo/matrix

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <set>
#include <complex>

using namespace std;

template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
class Matrix 
{
private:
    int mRows;
    int mCols;
    std::vector<T> mData;

    void validateIndexes(int row, int col) const 
    {
        if (row < 0 or row >= mRows)
        {
            throw invalid_argument("Invalid row index (" + to_string(row) + "): should be between 0 and " + to_string(mRows - 1));
        }
        if (col < 0 or col >= mCols)
        {
            throw invalid_argument("Invalid column index (" + to_string(col) + "): should be between 0 and " + to_string(mCols - 1));
        }
    }

public:
    Matrix() 
    {
        mRows = mCols = 0;
    }

    Matrix(int dimension) 
    {
        Matrix(dimension, dimension);
    }

    Matrix(int rows, int cols) : mRows(rows), mCols(cols), mData(rows* cols) 
    {

    }

    Matrix(int rows, int cols, const vector<T>& data) : mRows(rows), mCols(cols) 
    {
        if (data.size() != rows * cols)
        {
            throw invalid_argument("Matrix dimension incompatible with its initializing vector.");
        }
        mData = data;
    }

    template<int N>
    Matrix(int rows, int cols, T(&data)[N]) 
    {
        if (N != rows * cols)
        {
            throw invalid_argument("Matrix dimension incompatible with its initializing vector.");
        }
        vector<T> v(data, data + N);
        Matrix(rows, cols, v);
    }

    friend Matrix operator+(const Matrix& m, double value) 
    {
        Matrix result(m.mRows, m.mCols);

#pragma omp parallel for collapse(2) schedule(runtime)
        for (int i = 0; i < m.mRows; i++) 
        {
            for (int j = 0; j < m.mCols; j++) 
            {
                result(i, j) = value + m(i, j);
            }
        }

        return result;
    }

    friend Matrix operator+(double value, const Matrix& m) 
    {
        return m + value;
    }

    friend Matrix operator-(const Matrix& m, double value) 
    {
        return m + (-value);
    }

    friend Matrix operator-(double value, const Matrix& m) 
    {
        return m - value;
    }

    friend Matrix operator*(const Matrix& m, double value) 
    {
        Matrix result(m.mRows, m.mCols);

#pragma omp parallel for collapse(2) schedule(runtime)
        for (int i = 0; i < m.mRows; i++) 
        {
            for (int j = 0; j < m.mCols; j++) 
            {
                result(i, j) = value * m(i, j);
            }
        }

        return result;
    }

    friend Matrix operator*(double value, const Matrix& m) 
    {
        return m * value;
    }

    friend Matrix operator/(const Matrix& m, double value) 
    {
        Matrix result(m.mRows, m.mCols);

#pragma omp parallel for collapse(2) schedule(runtime)
        for (int i = 0; i < m.mRows; i++) 
        {
            for (int j = 0; j < m.mCols; j++) 
            {
                result(i, j) = m(i, j) / value;
            }
        }

        return result;
    }

    friend Matrix operator/(double value, const Matrix& m) 
    {
        
        Matrix result(m.mRows, m.mCols);

#pragma omp parallel for collapse(2) schedule(runtime)
        for (int i = 0; i < m.mRows; i++) 
        {
            for (int j = 0; j < m.mCols; j++) 
            {
                result(i, j) = value / m(i, j);
            }
        }

        return result;
    }

    Matrix operator+=(double value) 
    {
#pragma omp parallel for schedule(runtime)
        for (int i = 0; i < mData.size(); i++)
        {
            mData[i] += value;
        }
        return *this;
    }

    Matrix operator-=(double value) 
    {
#pragma omp parallel for schedule(runtime)
        for (int i = 0; i < mData.size(); i++)
        {
            mData[i] -= value;
        }
        return *this;
    }

    Matrix operator*=(double value) 
    {
#pragma omp parallel for schedule(runtime)
        for (int i = 0; i < mData.size(); i++)
        {
            mData[i] *= value;
        }
        return *this;
    }

    Matrix operator/=(double value) 
    {
#pragma omp parallel for schedule(runtime)
        for (int i = 0; i < mData.size(); i++)
        {
            mData[i] /= value;
        }
        return *this;
    }
    
    Matrix operator+(const Matrix& b) 
    {
        if (mRows != b.mRows || mCols != b.mCols)
        {
            throw invalid_argument("Cannot add these matrices: L = " + to_string(mRows) + "x" + to_string(mCols) + ", R = " + to_string(b.mRows) + "x" + to_string(b.mCols));
        }

        Matrix result(mRows, mCols);

#pragma omp parallel for collapse(2) schedule(runtime)
        for (int i = 0; i < mRows; i++) 
        {
            for (int j = 0; j < mCols; j++) 
            {
                result(i, j) = operator()(i, j) + b(i, j);
            }
        }

        return result;
    }

    Matrix operator-(const Matrix& b) 
    {
        if (mRows != b.mRows || mCols != b.mCols)
        {
            throw invalid_argument("Cannot subtract these matrices: L = " + to_string(mRows) + "x" + to_string(mCols) + ", R = " + to_string(b.mRows) + "x" + to_string(b.mCols));
        }

        Matrix result(mRows, mCols);

#pragma omp parallel for collapse(2) schedule(runtime)
        for (int i = 0; i < mRows; i++) 
        {
            for (int j = 0; j < mCols; j++) 
            {
                result(i, j) = operator()(i, j) - b(i, j);
            }
        }

        return result;
    }

    Matrix operator*(const Matrix& b) const 
    {
        if (mCols != b.mRows)
        {
            throw invalid_argument("Cannot multiply these matrices: L = " + to_string(this->mRows) + "x" + to_string(this->mCols) + ", R = " + to_string(b.mRows) + "x" + to_string(b.mCols));
        }

        Matrix result = zeros(mRows, b.mCols);

#pragma omp parallel for schedule(runtime) if(result.mRows * result.mCols > 250)
        for (int i = 0; i < result.mRows; i++) 
        {
            for (int k = 0; k < mCols; k++) 
            {
                double tmp = operator()(i, k);

                for (int j = 0; j < result.mCols; j++) 
                {
                    result(i, j) += tmp * b(k, j);
                }
            }
        }

        return result;
    }

    Matrix& operator+=(const Matrix& other) 
    {
        if (mRows != other.mRows || mCols != other.mCols)
        {
            throw invalid_argument("Cannot add these matrices: L = " + to_string(mRows) + "x" + to_string(mCols) + ", R = " + to_string(other.mRows) + "x" + to_string(other.mCols));
        }

#pragma omp parallel for collapse(2) schedule(runtime)
        for (int i = 0; i < other.mRows; i++) 
        {
            for (int j = 0; j < other.mCols; j++) 
            {
                operator()(i, j) += other(i, j);
            }
        }

        return *this;
    }

    Matrix& operator-=(const Matrix& other) 
    {
        if (mRows != other.mRows || mCols != other.mCols)
        {
            throw invalid_argument("Cannot subtract these matrices: L = " + to_string(mRows) + "x" + to_string(mCols) + ", R = " + to_string(other.mRows) + "x" + to_string(other.mCols));
        }

#pragma omp parallel for collapse(2) schedule(runtime)
        for (int i = 0; i < other.mRows; i++) 
        {
            for (int j = 0; j < other.mCols; j++) 
            {
                operator()(i, j) -= other(i, j);
            }
        }

        return *this;
    }

    Matrix& operator*=(const Matrix& other) {
        if (mCols != other.mRows)
        {
            throw invalid_argument("Cannot multiply these matrices: L " + to_string(mRows) + "x" + to_string(mCols) + ", R " + to_string(other.mRows) + "x" + to_string(other.mCols));
        }

        Matrix result(mRows, other.mCols);

#pragma omp parallel for collapse(2) schedule(runtime)
        
        for (int i = 0; i < result.mRows; i++) 
        {
            for (int j = 0; j < result.mCols; j++) 
            {
                
                result(i, j) = 0;

                for (int ii = 0; ii < mCols; ii++)
                {
                    result(i, j) += operator()(i, ii) * other(ii, j);
                }
            }
        }

        mRows = result.mRows;
        mCols = result.mCols;
        mData = result.mData;

        return *this;
    }

    Matrix<int> operator==(const T& value) 
    {
        Matrix<int> result(mRows, mCols);

#pragma omp parallel for collapse(2) schedule(runtime)
        for (int i = 0; i < mRows; i++) 
        {
            for (int j = 0; j < mCols; j++) 
            {
                result(i, j) = operator()(i, j) == value;
            }
        }

        return result;
    }

    bool operator==(const Matrix& other) 
    {
        if (mData.size() != other.mData.size() || mRows != other.mRows || mCols != other.mCols)
        {
            return false;
        }

        for (int k = 0; k < mData.size(); k++) 
        {
            if (mData[k] != other.mData[k])
            {
                return false;
            }
        }

        return true;
    }

    Matrix operator!=(const double& value) 
    {
        
        
        return -((*this == value) - 1);
    }

    bool operator!=(const Matrix& other) 
    {
        
        
        return !(*this == other);
    }

    Matrix operator-() 
    {
        Matrix result(this->mRows, this->mCols);

#pragma omp parallel for collapse(2) schedule(runtime)
        for (int i = 0; i < mCols; i++) 
        {
            for (int j = 0; j < mRows; j++) 
            {
                result(i, j) = -operator()(i, j);
            }
        }

        return result;
    }

    T& operator()(int i, int j) 
    {
        validateIndexes(i, j);
        return mData[i * mCols + j];
    }

    T operator()(int i, int j) const 
    {
        validateIndexes(i, j);
        return mData[i * mCols + j];
    }

    static Matrix fill(int rows, int cols, double value) 
    {
        Matrix result(rows, cols, vector<T>(rows * cols, value));
        return result;
    }

    bool isSquare() const 
    {
        return mCols == mRows;
    }

    static Matrix zeros(int rows, int cols) 
    {
        return fill(rows, cols, 0);
    }

    Matrix submatrix(int row, int column) const 
    {
        Matrix result(mRows - 1, mCols - 1);

        int subi = 0;

#pragma omp parallel for schedule(runtime)
        for (int i = 0; i < mRows; i++) 
        {
            int subj = 0;

            if (i == row) 
            {
                continue;
            }

            for (int j = 0; j < mCols; j++) 
            {
                if (j == column) 
                {
                    continue;
                }

                result(subi, subj) = operator()(i, j);
                subj++;
            }
            subi++;
        }

        return result;
    }

    
    
    
    
    
    double getMinor(int row, int column) const 
    {
        
        
        if (mRows == 2 and mCols == 2) 
        {
            Matrix result(2, 2);

            result(0, 0) = operator()(1, 1);
            result(0, 1) = operator()(1, 0);
            result(1, 0) = operator()(0, 1);
            result(1, 1) = operator()(0, 0);

            return result.determinant();
        }

        return submatrix(row, column).determinant();
    }

    
    
    
    
    double cofactor(int row, int column) const 
    {
        double minor;

        
        if (mRows == 2 and mCols == 2) 
        {
            if (row == 0 and column == 0)
            {
                minor = operator()(1, 1);
            }
            else if (row == 1 and column == 1)
            {
                minor = operator()(0, 0);
            }
            else if (row == 0 and column == 1)
            {
                minor = operator()(1, 0);
            }
            else if (row == 1 and column == 0)
            {
                minor = operator()(0, 1);
            }
        }
        else
        {
            minor = this->getMinor(row, column);
        }

        return (row + column) % 2 == 0 ? minor : -minor;
    }

    
    
    Matrix cofactorMatrix() const 
    {
        Matrix result(mRows, mCols);

#pragma omp parallel for collapse(2) schedule(runtime)
        for (int i = 0; i < mRows; i++) 
        {
            for (int j = 0; j < mCols; j++) 
            {
                result(i, j) = cofactor(i, j);
            }
        }

        return result;
    }

    
    
    Matrix adjugate() const 
    {
        return cofactorMatrix().transpose();
    }

    
    
    
    Matrix inverse() const 
    {
        if (!isSquare())
        {
            throw runtime_error("Cannot invert a non-square matrix");
        }

        double det = determinant();

        if (det == 0)
        {
            throw runtime_error("Matrix is singular");
        }

        Matrix adj = adjugate();

        return adjugate() / det;
    };

    
    
    double determinant() const 
    {
        if (!isSquare()) 
        {
            throw runtime_error("Cannot calculate the determinant of a non-square matrix");
        }

        int n = mRows;
        double d = 0;

        if (n == 2) 
        {
            return ((operator()(0, 0) * operator()(1, 1)) - (operator()(1, 0) * operator()(0, 1)));
        }
        else 
        {
#pragma omp parallel for reduction (+:d)
            for (int c = 0; c < n; c++) 
            {
                d += pow(-1, c) * operator()(0, c) * submatrix(0, c).determinant();
            }

            return d;
        }
    }

    
    
    Matrix transpose() const 
    {
        Matrix result(mCols, mRows);

#pragma omp parallel for collapse(2) schedule(runtime)
        for (int i = 0; i < mRows; i++) 
        {
            for (int j = 0; j < mCols; j++) 
            {
                result(j, i) = operator()(i, j);
            }
        }

        return result;
    }

    
    
    
    
    friend ostream& operator<<(ostream& os, const Matrix& matrix) 
    {
        const int numWidth = 13;
        char fill = ' ';

        for (int i = 0; i < matrix.mRows; i++) 
        {
            for (int j = 0; j < matrix.mCols; j++) 
            {
                os << left << setw(numWidth) << setfill(fill) << to_string(matrix(i, j));
            }
            os << endl;
        }

        return os;
    }

    
    
    Matrix copy() 
    {
        Matrix result(mRows, mCols);
        result.mData = mData;

        return result;
    }

    T min() const 
    {
        return *std::min_element(std::begin(mData), std::end(mData));
    }

    T max() const 
    {
        return *std::max_element(std::begin(mData), std::end(mData));
    }

    Matrix<T> apply(function<T(T)> f) 
    {
        Matrix<T> result(mRows, mCols, vector<T>(mRows * mCols, 0));
        std::transform(mData.begin(), mData.end(), result.mData.begin(), f);

        return result;
    }
};

typedef Matrix<double> MatrixD;
typedef Matrix<int> MatrixI;

#endif 