// https://github.com/douglasrizzo/matrix

#ifndef MATRIX_S_HPP
#define MATRIX_S_HPP

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

//! Matrix_S implementation, with a series of linear algebra functions
//! @tparam T The arithmetic type the matrix will store
template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
class Matrix_S 
{
private:
    int mRows;
    int mCols;
    std::vector<T> mData;

    //! Validates if indices are contained inside the matrix
    //! \param row row index
    //! \param col column index
    //! \throws runtime error if at least one of the indices is out of bounds
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
    //region Constructors

    //! Initializes an empty matrix
    Matrix_S() 
    {
        mRows = mCols = 0;
    }

    //! Initializes a square matrix
    //! \param dimension number of rows and columns
    Matrix_S(int dimension) 
    {
        Matrix_S(dimension, dimension);
    }

    //! Initializes a matrix with a predetermined number of rows and columns
    //! \param rows number of rows in the matrix
    //! \param cols number of columns in the matrix
    Matrix_S(int rows, int cols) : mRows(rows), mCols(cols), mData(rows* cols) 
    {

    }

    //! Initializes a matrix with a predetermined number of rows and columns and populates it with data
    //! \param rows number of rows in the matrix
    //! \param cols number of columns in the matrix
    //! \param data a vector containing <code>rows * cols</code> elements to populate the matrix
    Matrix_S(int rows, int cols, const vector<T>& data) : mRows(rows), mCols(cols) 
    {
        if (data.size() != rows * cols)
        {
            throw invalid_argument("Matrix_S dimension incompatible with its initializing vector.");
        }
        mData = data;
    }

    template<int N>
    Matrix_S(int rows, int cols, T(&data)[N]) 
    {
        if (N != rows * cols)
        {
            throw invalid_argument("Matrix_S dimension incompatible with its initializing vector.");
        }
        vector<T> v(data, data + N);
        Matrix_S(rows, cols, v);
    }
    //endregion

    //region Operators

    //region Scalar operators

    //! Scalar addition
    //! \param m a matrix
    //! \param value scalar to be added to the matrix
    //! \return the result of the scalar addition of <code>m</code> and <code>value</code>
    friend Matrix_S operator+(const Matrix_S& m, double value) 
    {
        Matrix_S result(m.mRows, m.mCols);

        for (int i = 0; i < m.mRows; i++) 
        {
            for (int j = 0; j < m.mCols; j++) 
            {
                result(i, j) = value + m(i, j);
            }
        }

        return result;
    }

    //! Scalar addition
    //! \param m a matrix
    //! \param value scalar to be added to the matrix
    //! \return the result of the scalar addition of <code>m</code> and <code>value</code>
    friend Matrix_S operator+(double value, const Matrix_S& m) 
    {
        return m + value;
    }

    //! Scalar subtraction
    //! \param m a matrix
    //! \param value scalar to be subtracted to the matrix
    //! \return the result of the scalar subtraction of <code>m</code> and <code>value</code>
    friend Matrix_S operator-(const Matrix_S& m, double value) 
    {
        return m + (-value);
    }

    //! Scalar subtraction
    //! \param m a matrix
    //! \param value scalar to be subtracted to the matrix
    //! \return the result of the scalar subtraction of <code>m</code> and <code>value</code>
    friend Matrix_S operator-(double value, const Matrix_S& m) 
    {
        return m - value;
    }

    //! Scalar multiplication
    //! \param m a matrix
    //! \param value scalar to be multiplied by the matrix
    //! \return the result of the scalar multiplication of <code>m</code> and <code>value</code>
    friend Matrix_S operator*(const Matrix_S& m, double value) 
    {
        Matrix_S result(m.mRows, m.mCols);

        for (int i = 0; i < m.mRows; i++) 
        {
            for (int j = 0; j < m.mCols; j++) 
            {
                result(i, j) = value * m(i, j);
            }
        }

        return result;
    }

    //! Scalar multiplication
    //! \param m a matrix
    //! \param value scalar to be multiplied by the matrix
    //! \return the result of the scalar multiplication of <code>m</code> and <code>value</code>
    friend Matrix_S operator*(double value, const Matrix_S& m) 
    {
        return m * value;
    }

    //! Scalar division
    //! \param m a matrix
    //! \param value scalar to be divide the matrix by
    //! \return the result of the scalar division of <code>m</code> by <code>value</code>
    friend Matrix_S operator/(const Matrix_S& m, double value) 
    {
        Matrix_S result(m.mRows, m.mCols);

        for (int i = 0; i < m.mRows; i++) 
        {
            for (int j = 0; j < m.mCols; j++) 
            {
                result(i, j) = m(i, j) / value;
            }
        }

        return result;
    }

    //! Scalar division
    //! \param value scalar that will be divided by the matrix
    //! \param m a matrix
    //! \return the result of the scalar division of <code>value</code> by <code>m</code>
    friend Matrix_S operator/(double value, const Matrix_S& m) 
    {
        // division is not commutative, so a new method is implemented
        Matrix_S result(m.mRows, m.mCols);

        for (int i = 0; i < m.mRows; i++) 
        {
            for (int j = 0; j < m.mCols; j++) 
            {
                result(i, j) = value / m(i, j);
            }
        }

        return result;
    }

    Matrix_S operator+=(double value) 
    {
        for (int i = 0; i < mData.size(); i++)
        {
            mData[i] += value;
        }
        return *this;
    }

    Matrix_S operator-=(double value) 
    {
        for (int i = 0; i < mData.size(); i++)
        {
            mData[i] -= value;
        }
        return *this;
    }

    Matrix_S operator*=(double value) 
    {
        for (int i = 0; i < mData.size(); i++)
        {
            mData[i] *= value;
        }
        return *this;
    }

    Matrix_S operator/=(double value) 
    {
        for (int i = 0; i < mData.size(); i++)
        {
            mData[i] /= value;
        }
        return *this;
    }
    //endregion

    //region Matrix_S operators

    //! Matrix_S addition operation
    //! \param b another matrix
    //! \return Result of the addition of both matrices
    Matrix_S operator+(const Matrix_S& b) 
    {
        if (mRows != b.mRows || mCols != b.mCols)
        {
            throw invalid_argument("Cannot add these matrices: L = " + to_string(mRows) + "x" + to_string(mCols) + ", R = " + to_string(b.mRows) + "x" + to_string(b.mCols));
        }

        Matrix_S result(mRows, mCols);

        for (int i = 0; i < mRows; i++) 
        {
            for (int j = 0; j < mCols; j++) 
            {
                result(i, j) = operator()(i, j) + b(i, j);
            }
        }

        return result;
    }

    //! Matrix_S subtraction operation
    //! \param b another matrix
    //! \return Result of the subtraction of both matrices
    Matrix_S operator-(const Matrix_S& b) 
    {
        if (mRows != b.mRows || mCols != b.mCols)
        {
            throw invalid_argument("Cannot subtract these matrices: L = " + to_string(mRows) + "x" + to_string(mCols) + ", R = " + to_string(b.mRows) + "x" + to_string(b.mCols));
        }

        Matrix_S result(mRows, mCols);

        for (int i = 0; i < mRows; i++) 
        {
            for (int j = 0; j < mCols; j++) 
            {
                result(i, j) = operator()(i, j) - b(i, j);
            }
        }

        return result;
    }

    //! Matrix_S multiplication operation
    //! \param b another matrix
    //! \return Result of the multiplication of both matrices
    Matrix_S operator*(const Matrix_S& b) const 
    {
        if (mCols != b.mRows)
        {
            throw invalid_argument("Cannot multiply these matrices: L = " + to_string(this->mRows) + "x" + to_string(this->mCols) + ", R = " + to_string(b.mRows) + "x" + to_string(b.mCols));
        }

        Matrix_S result = zeros(mRows, b.mCols);

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

    Matrix_S& operator+=(const Matrix_S& other) 
    {
        if (mRows != other.mRows || mCols != other.mCols)
        {
            throw invalid_argument("Cannot add these matrices: L = " + to_string(mRows) + "x" + to_string(mCols) + ", R = " + to_string(other.mRows) + "x" + to_string(other.mCols));
        }

        for (int i = 0; i < other.mRows; i++) 
        {
            for (int j = 0; j < other.mCols; j++) 
            {
                operator()(i, j) += other(i, j);
            }
        }

        return *this;
    }

    Matrix_S& operator-=(const Matrix_S& other) 
    {
        if (mRows != other.mRows || mCols != other.mCols)
        {
            throw invalid_argument("Cannot subtract these matrices: L = " + to_string(mRows) + "x" + to_string(mCols) + ", R = " + to_string(other.mRows) + "x" + to_string(other.mCols));
        }

        for (int i = 0; i < other.mRows; i++) 
        {
            for (int j = 0; j < other.mCols; j++) 
            {
                operator()(i, j) -= other(i, j);
            }
        }

        return *this;
    }

    Matrix_S& operator*=(const Matrix_S& other) {
        if (mCols != other.mRows)
        {
            throw invalid_argument("Cannot multiply these matrices: L " + to_string(mRows) + "x" + to_string(mCols) + ", R " + to_string(other.mRows) + "x" + to_string(other.mCols));
        }

        Matrix_S result(mRows, other.mCols);

        // two loops iterate through every cell of the new matrix
        for (int i = 0; i < result.mRows; i++) 
        {
            for (int j = 0; j < result.mCols; j++) 
            {
                // here we calculate the value of a single cell in our new matrix
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
    //endregion

    //region Equality operators

    Matrix_S<int> operator==(const T& value) 
    {
        Matrix_S<int> result(mRows, mCols);

        for (int i = 0; i < mRows; i++) 
        {
            for (int j = 0; j < mCols; j++) 
            {
                result(i, j) = operator()(i, j) == value;
            }
        }

        return result;
    }

    bool operator==(const Matrix_S& other) 
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

    Matrix_S operator!=(const double& value) 
    {
        // subtract 1 from everything: 0s become -1s, 1s become 0s
        // negate everything: 0s remains 0s, -1s becomes 1s
        return -((*this == value) - 1);
    }

    bool operator!=(const Matrix_S& other) 
    {
        // subtract 1 from everything: 0s become -1s, 1s become 0s
        // negate everything: 0s remains 0s, -1s becomes 1s
        return !(*this == other);
    }
    //endregion

    //! Matrix_S negative operation
    //! \return The negative of the current matrix
    Matrix_S operator-() 
    {
        Matrix_S result(this->mRows, this->mCols);

        for (int i = 0; i < mCols; i++) 
        {
            for (int j = 0; j < mRows; j++) 
            {
                result(i, j) = -operator()(i, j);
            }
        }

        return result;
    }

    //region Functors

    //! Functor used to access elements in the matrix
    //! \param i row index
    //! \param j column index
    //! \return element in position ij of the matrix
    T& operator()(int i, int j) 
    {
        validateIndexes(i, j);
        return mData[i * mCols + j];
    }

    //! Functor used to access elements in the matrix
    //! \param i row index
    //! \param j column index
    //! \return element in position ij of the matrix
    T operator()(int i, int j) const 
    {
        validateIndexes(i, j);
        return mData[i * mCols + j];
    }

    //! Returns a matrix filled with a single value
    //! \param rows number of rows in the matrix
    //! \param cols number of columns in the matrix
    //! \param value value to be used for initialization
    //! \return a matrix with all values set to <code>value</code>
    static Matrix_S fill(int rows, int cols, double value) 
    {
        Matrix_S result(rows, cols, vector<T>(rows * cols, value));
        return result;
    }

    bool isSquare() const 
    {
        return mCols == mRows;
    }

    //! Returns a matrix filled with zeros
    //! \param rows number of rows in the matrix
    //! \param cols number of columns in the matrix
    //! \return matrix filled with zeros
    static Matrix_S zeros(int rows, int cols) 
    {
        return fill(rows, cols, 0);
    }

    //! Returns a submatrix of the current matrix, removing one row and column of the original matrix
    //! \param row index of the row to be removed
    //! \param column index of the column to be removed
    //! \return submatrix of the current matrix, with one less row and column
    Matrix_S submatrix(int row, int column) const 
    {
        Matrix_S result(mRows - 1, mCols - 1);

        int subi = 0;

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

    //! Returns the minor of a matrix, which is the determinant of a submatrix
    //! where a single row and column are removed
    //! \param row index of the row to be removed
    //! \param column index of the column to be removed
    //! \return minor of the current matrix
    double getMinor(int row, int column) const 
    {
        //        the minor of a 2x2 a b is d c
        //                           c d    b a
        if (mRows == 2 and mCols == 2) 
        {
            Matrix_S result(2, 2);

            result(0, 0) = operator()(1, 1);
            result(0, 1) = operator()(1, 0);
            result(1, 0) = operator()(0, 1);
            result(1, 1) = operator()(0, 0);

            return result.determinant();
        }

        return submatrix(row, column).determinant();
    }

    //! Calculates the cofactor of a matrix at a given point
    //! \param row index of the row where the cofactor will be calculated
    //! \param column index of the column where the cofactor will be calculated
    //! \return cofactor of the matrix at the given position
    double cofactor(int row, int column) const 
    {
        double minor;

        // special case for when our matrix is 2x2
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

    //! Calculates the cofactor matrix
    //! \return Cofactor matrix of the current matrix
    Matrix_S cofactorMatrix() const 
    {
        Matrix_S result(mRows, mCols);

        for (int i = 0; i < mRows; i++) 
        {
            for (int j = 0; j < mCols; j++) 
            {
                result(i, j) = cofactor(i, j);
            }
        }

        return result;
    }

    //! Returns the adjugate of the current matrix, which is the transpose of its cofactor matrix
    //! \return Adjugate of the current matrix
    Matrix_S adjugate() const 
    {
        return cofactorMatrix().transpose();
    }

    //! Calculates the inverse of the current matrix. Raises an error if
    //! the matrix is singular, that is, its determinant is equal to 0
    //! \return inverse of the current matrix
    Matrix_S inverse() const 
    {
        if (!isSquare())
        {
            throw runtime_error("Cannot invert a non-square matrix");
        }

        double det = determinant();

        if (det == 0)
        {
            throw runtime_error("Matrix_S is singular");
        }

        Matrix_S adj = adjugate();

        return adjugate() / det;
    };

    //! Calculates the determinant of the matrix
    //! \return determinant of the matrix
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
            for (int c = 0; c < n; c++) 
            {
                d += pow(-1, c) * operator()(0, c) * submatrix(0, c).determinant();
            }

            return d;
        }
    }

    //! Returns the transpose of a matrix
    //! \return transpose of the current matrix
    Matrix_S transpose() const 
    {
        Matrix_S result(mCols, mRows);

        for (int i = 0; i < mRows; i++) 
        {
            for (int j = 0; j < mCols; j++) 
            {
                result(j, i) = operator()(i, j);
            }
        }

        return result;
    }

    //! Prints a matrix
    //! \param os output stream
    //! \param matrix the matrix to be printed
    //! \return output stream with the string representation of the matrix
    friend ostream& operator<<(ostream& os, const Matrix_S& matrix) 
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

    //! Returns a copy of the matrix
    //! \return copy of the current matrix
    Matrix_S copy() 
    {
        Matrix_S result(mRows, mCols);
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

    Matrix_S<T> apply(function<T(T)> f) 
    {
        Matrix_S<T> result(mRows, mCols, vector<T>(mRows * mCols, 0));
        std::transform(mData.begin(), mData.end(), result.mData.begin(), f);

        return result;
    }
};

typedef Matrix_S<double> MatrixD_S;
typedef Matrix_S<int> MatrixI_S;

#endif //MATRIX_S_HPP