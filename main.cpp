#define _OPENMP_LLVM_RUNTIME
#include <omp.h>
#include <iostream>
#include "matrix_single_threaded.hpp"
#include "matrix_helper_single_threaded.hpp"
#include "matrix_multi_threaded.hpp"
#include "matrix_helper_multi_threaded.hpp"
#include "text_table.hpp"

double sequentialAlgorithm(int N, bool verbose = false)
{
    omp_set_num_threads(1);
    omp_set_schedule(omp_sched_static, 0);

    MatrixHelper_S::setSeed(3);

    MatrixD_S A = MatrixHelper_S::randomNonZeroDeterminantMatrix(N);
    MatrixD_S B = MatrixHelper_S::randomMatrix(N, 1);

    double start = omp_get_wtime();

    double det = A.determinant();
    MatrixD_S A_adj = A.adjugate();
    MatrixD_S A_inv = A_adj / det;
    MatrixD_S X = A_inv * B;

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

    MatrixHelper_M::setSeed(3);

    MatrixD_M A = MatrixHelper_M::randomNonZeroDeterminantMatrix(N);
    MatrixD_M B = MatrixHelper_M::randomMatrix(N, 1);

    double start = omp_get_wtime();

    double det = A.determinant();
    MatrixD_M A_adj = A.adjugate();
    MatrixD_M A_inv = A_adj / det;
    MatrixD_M X = A_inv * B;

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

// void testAlgorithms()
// {
//     std::vector<int> Ns { 5, 7, 9 };
//     std::vector<int> Ms { 2, 4, 8, 16 };

//     auto printSeq = [&](std::vector<double> table) {
//         TextTable t( '-', '|', '+' );

//         t.add( "N" );
//         t.add( "t" );
//         t.endOfRow();

//         for (size_t i = 0; i < Ns.size(); ++i)
//         {
//             t.add( std::to_string(Ns[i]) );
//             t.add( std::to_string(table[i]) );
//             t.endOfRow();
//         }

//         std::cout << t;
//     };

//     auto printMul = [&](std::vector<std::vector<double>> table) {
//         TextTable t( '-', '|', '+' );

//         t.add( "N \\ M" );

//         for (size_t j = 0; j < Ms.size(); ++j)
//         {
//             t.add( std::to_string(Ms[j]) );
//         }

//         t.endOfRow();

//         for (size_t i = 0; i < Ns.size(); ++i)
//         {
//             t.add( std::to_string(Ns[i]) );

//             for (size_t j = 0; j < Ms.size(); ++j)
//             {
//                 t.add( std::to_string(table[i][j]) );
//             }

//             t.endOfRow();
//         }

//         std::cout << t;
//     };

//     std::vector<double> seqValues;

//     for (size_t i = 0; i < Ns.size(); ++i)
//     {
//         int N = Ns[i];

//         double value = sequentialAlgorithm(N);

//         seqValues.push_back(value);
//     }

//     printSeq(seqValues);

//     std::vector<std::vector<double>> mulValues(Ns.size());

//     for (size_t i = 0; i < Ns.size(); ++i)
//     {
//         for (size_t j = 0; j < Ms.size(); ++j)
//         {
//             int N = Ns[i];
//             int M = Ms[j];

//             double value = multiThreadedAlgorithm(N, M);

//             mulValues[i].push_back(value);
//         }
//     }

//     printMul(mulValues);
// }


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
            t.add( std::to_string( seqValues[j] ) );
            t.add( std::to_string( mulValues[i][j] ) );
            t.add( std::to_string( seqValues[j] / (Ms[j] * mulValues[i][j]) ) );
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