#define _OPENMP_LLVM_RUNTIME
#include <omp.h>
#include <iostream>
#include "matrix.hpp"
#include "matrix_helper.hpp"
#include "text_table.hpp"

double sequentialAlgorithm(int N, bool verbose = false)
{
    omp_set_num_threads(1);
    omp_set_schedule(omp_sched_static, 0);

    MatrixHelper::setSeed(3);

    MatrixD A = MatrixHelper::randomNonZeroDeterminantMatrix(N);
    MatrixD B = MatrixHelper::randomMatrix(N, 1);

    double start = omp_get_wtime();

    double det = A.determinant();
    MatrixD A_adj = A.adjugate();
    MatrixD A_inv = A_adj / det;
    MatrixD X = A_inv * B;

    double end = omp_get_wtime();

    if (verbose)
    {
        std::cout << "A = \n" << A << std::endl;
        std::cout << "B = \n" << B << std::endl;
        std::cout << "X = \n" << X << std::endl;
        std::cout << "B = A * X = \n" << A * X << std::endl;
    }

    double elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed << "s." << std::endl;

    return elapsed;
}

double multiThreadedAlgorithm(int N, int M, bool useStatic = true, bool verbose = false)
{
    omp_set_num_threads(M);

    if (useStatic)
    {
        omp_set_schedule(omp_sched_static, 0);
    }
    else
    {
        omp_set_schedule(omp_sched_dynamic, 0);
    }

    MatrixHelper::setSeed(3);

    MatrixD A = MatrixHelper::randomNonZeroDeterminantMatrix(N);
    MatrixD B = MatrixHelper::randomMatrix(N, 1);

    double start = omp_get_wtime();

    double det = A.determinant();
    MatrixD A_adj = A.adjugate();
    MatrixD A_inv = A_adj / det;
    MatrixD X = A_inv * B;

    double end = omp_get_wtime();

    if (verbose)
    {
        std::cout << "A = \n" << A << std::endl;
        std::cout << "B = \n" << B << std::endl;
        std::cout << "X = \n" << X << std::endl;
        std::cout << "B = A * X = \n" << A * X << std::endl;
    }

    double elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed << "s." << std::endl;

    return elapsed;
}

void testAlgorithms(bool useStatic = true)
{
    TextTable t( '-', '|', '+' );

    t.add( "N" );
    t.add( "M" );
    t.add( "T_1" );
    t.add( "T_p" );
    t.add( "E_p" );
    t.endOfRow();

    std::vector<int> Ns { 6, 7, 8 };
    std::vector<int> Ms { 2, 4, 8, 16 };

    std::vector<double> seqValues;

    for (size_t i = 0; i < Ns.size(); ++i)
    {
        int N = Ns[i];

        double value = sequentialAlgorithm(N);

        seqValues.push_back(value);
    }


    std::vector<std::vector<double>> mulValues(Ns.size());

    for (int i = 0; i < Ns.size(); ++i)
    {
        for (int j = 0; j < Ms.size(); ++j)
        {
            int N = Ns[i];
            int M = Ms[j];

            double value = multiThreadedAlgorithm(N, M, useStatic);

            mulValues[i].push_back(value);
        }
    }

    for (int i = 0; i < Ns.size(); ++i)
    {
        for (int j = 0; j < Ms.size(); ++j)
        {
            t.add( std::to_string( Ns[i] ) );
            t.add( std::to_string( Ms[j] ) );
            t.add( std::to_string( seqValues[i] ) );
            t.add( std::to_string( mulValues[i][j] ) );
            t.add( std::to_string( seqValues[i] / (Ms[j] * mulValues[i][j]) ) );
            t.endOfRow();
        }
    }

    std::cout << t;
}

void testAlgorithmsSchedules()
{
    testAlgorithms(true);
    testAlgorithms(false);
}

int main()
{
    sequentialAlgorithm(7, true);
    multiThreadedAlgorithm(7, 8, true, true);

    testAlgorithms();

    testAlgorithmsSchedules();

    return 0;
}