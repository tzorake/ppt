#define _OPENMP_LLVM_RUNTIME
#include <omp.h>
#include <iostream>
#include "matrix.hpp"
#include "matrix_helper.hpp"
#include "text_table.hpp"

double measure_single(int N, bool verbose = false)
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

    if (verbose) {
        std::cout << "A = \n" << A << std::endl;
        std::cout << "B = \n" << B << std::endl;
        std::cout << "X = \n" << X << std::endl;
        std::cout << "B = A * X = \n" << A * X << std::endl;
    }

    double elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed << "s." << std::endl;

    return elapsed;
}

double measure_multi(int N, int M, bool useStatic = true, bool verbose = false)
{
    omp_set_num_threads(M);

    if (useStatic) {
        omp_set_schedule(omp_sched_static, 0);
    }
    else {
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

    if (verbose) {
        std::cout << "A = \n" << A << std::endl;
        std::cout << "B = \n" << B << std::endl;
        std::cout << "X = \n" << X << std::endl;
        std::cout << "B = A * X = \n" << A * X << std::endl;
    }

    double elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed << "s." << std::endl;

    return elapsed;
}

double measure_multi_complex(int N, int M, int k, bool useStatic = true, bool verbose = false)
{
    omp_set_num_threads(M);

    if (useStatic) {
        omp_set_schedule(omp_sched_static, 0);
    }
    else {
        omp_set_schedule(omp_sched_dynamic, 0);
    }

    MatrixHelper::setSeed(3);

    MatrixD A = MatrixHelper::randomNonZeroDeterminantMatrix(N);
    MatrixD B = MatrixHelper::randomMatrix(N, 1);

    MatrixD C = MatrixHelper::randomNonZeroDeterminantMatrix(k);

    double start = omp_get_wtime();

    double A_det = A.determinant();
    MatrixD A_adj = A.adjugate();
    MatrixD A_inv = A_adj / A_det;

    for (int i = 0; i < k; ++i) {
        MatrixD D = C.copy();
        C *= D * D.adjugate() / D.determinant();
    }

    MatrixD X = A_inv * B;

    double end = omp_get_wtime();

    if (verbose) {
        std::cout << "A = \n" << A << std::endl;
        std::cout << "B = \n" << B << std::endl;
        std::cout << "X = \n" << X << std::endl;
        std::cout << "B = A * X = \n" << A * X << std::endl;
    }

    double elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed << "s." << std::endl;

    return elapsed;
}

void measure_using_schedule(bool useStatic = true)
{
    TextTable t( '-', '|', '+' );

    t.add( "N" );
    t.add( "M" );
    t.add( "T_1" );
    t.add( "T_p" );
    t.add( "E_p" );
    t.endOfRow();

    std::vector<int> Ns { 8, 9, 10 };
    std::vector<int> Ms { 2, 3, 4, 5 };

    std::vector<double> seqValues;

    for (size_t i = 0; i < Ns.size(); ++i) {
        int N = Ns[i];

        double value = measure_single(N);

        seqValues.push_back(value);
    }

    std::vector<std::vector<double>> mulValues(Ns.size());

    for (int i = 0; i < Ns.size(); ++i) {
        for (int j = 0; j < Ms.size(); ++j) {
            int N = Ns[i];
            int M = Ms[j];

            double value = measure_multi(N, M, useStatic);

            mulValues[i].push_back(value);
        }
    }

    for (int i = 0; i < Ns.size(); ++i) {
        for (int j = 0; j < Ms.size(); ++j) {
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

void measure_using_schedule_complex()
{
    TextTable t( '-', '|', '+' );

    t.add( "N" );
    t.add( "M" );
    t.add( "K" );
    t.add( "T_1" );
    t.add( "T_p" );
    t.add( "E_p" );
    t.endOfRow();

    std::vector<int> Ns { 8, 9, 10 };
    std::vector<int> Ms { 2, 3, 4, 5 };
    std::vector<int> Ks { 8, 9 };

    std::vector<double> seqValues;

    for (size_t i = 0; i < Ns.size(); ++i) {
        int N = Ns[i];

        double value = measure_single(N);

        seqValues.push_back(value);
    }

    std::vector<std::vector<double>> mulValues(Ns.size());

    for (int i = 0; i < Ns.size(); ++i) {
        for (int j = 0; j < Ms.size(); ++j) {
            for (int k = 0; k < Ks.size(); ++k) {
                    int N = Ns[i];
                    int M = Ms[j];
                    int K = Ks[k];

                    double value = measure_multi_complex(N, M, K);

                    mulValues[i].push_back(value);
            }
        }
    }

    for (int i = 0; i < Ns.size(); ++i) {
        for (int j = 0; j < Ms.size(); ++j) {
            for (int k = 0; k < Ks.size(); ++k) {
                t.add( std::to_string( Ns[i] ) );
                t.add( std::to_string( Ms[j] ) );
                t.add( std::to_string( Ks[k] ) );
                t.add( std::to_string( seqValues[i] ) );
                t.add( std::to_string( mulValues[i][j*Ks.size() + k] ) );
                t.add( std::to_string( seqValues[i] / (Ms[j] * mulValues[i][j + k]) ) );
                t.endOfRow();
            }
        }
    }

    std::cout << t;
}

void compare_schedules()
{
    measure_using_schedule(true);
    measure_using_schedule(false);
}

int main()
{
    // UNCOMMENT IF WANT TO CHECK MEASUREMENT FOR SINGLETHREADED
    // measure_single(10, true);

    // UNCOMMENT IF WANT TO CHECK MEASUREMENT FOR MULTITHREADED
    // measure_multi(10, 4, true, true);

    // UNCOMMENT IF WANT TO CHECK MEASUREMENT COMPARISON FOR SINGLETHREADED AND MULTITHREADED
    // measure_using_schedule();

    // measure_using_schedule_complex();

    compare_schedules();

    return 0;
}