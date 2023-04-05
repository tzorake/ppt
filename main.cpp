#define _OPENMP_LLVM_RUNTIME
#include <omp.h>
#include <iostream>
#include "matrix.hpp"
#include "matrix_helper.hpp"
#include "text_table.hpp"

#define VERBOSE false

double solveRandomSystemOfLinearEquations(int count)
{
    MatrixHelper::setSeed(3);

    MatrixD A = MatrixHelper::getRandomNonZeroDeterminantMatrix(count);
    MatrixD B = MatrixHelper::getRandomMatrix(count, 1);

    double start = omp_get_wtime();

    double det = A.determinant();
    MatrixD A_adj = A.adjugate();
    MatrixD A_inv = 1.0 / det * A_adj;
    MatrixD X = A_inv * B;

    double end = omp_get_wtime();

#if VERBOSE == true
    std::cout << "A = \n" << A << std::endl;
    std::cout << "B = \n" << B << std::endl;
    std::cout << "X = \n" << X << std::endl;
    std::cout << "B = A * X = \n" << A * X << std::endl;
#endif

    double elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed << "s." << std::endl;

    return elapsed;
}

void testSystemOfLinearEquationsSolver()
{
    int N = 0;
    int M = 0;
    std::string temp;

    std::cout << "Enter count of linear equations in the system:" << std::endl;
    std::cin >> N;

    std::cout << "Enter count of threads:" << std::endl;
    std::cin >> M;

    std::cout << "Use static decomposition (Y/n)? (default: y)" << std::endl;
    std::cin >> temp;

    bool useStatic = (temp != "n");

    int count = N / M;

    if (useStatic)
    {
        omp_set_schedule(omp_sched_static, count > 0 ? count : 1);
        std::cout << "Static decomposition is being used" << std::endl;
    }
    else
    {
        omp_set_schedule(omp_sched_dynamic, count > 0 ? count : 1);
        std::cout << "Dynamic decomposition is being used" << std::endl;
    }

    omp_set_num_threads(1);
    solveRandomSystemOfLinearEquations(N);

    omp_set_num_threads(M);
    solveRandomSystemOfLinearEquations(N);
}

void table()
{
    std::vector<int> Ns { 5, 7, 9 };
    std::vector<int> Ms { 2, 4, 8, 16 };

    std::vector<std::vector<double>> values(Ns.size());

    for (size_t i = 0; i < Ns.size(); ++i)
    {
        for (size_t j = 0; j < Ms.size(); ++j)
        {
            int N = Ns[i];
            int M = Ms[j];

            std::cout << " ***** [N = " << N << ", M = " << M << "] *****" << std::endl;

            omp_set_num_threads(M);
            double value = solveRandomSystemOfLinearEquations(N);

            values[i].push_back(value);
        }
    }

    TextTable t( '-', '|', '+' );

    t.add( "N \\ M" );

    for (size_t j = 0; j < Ms.size(); ++j)
    {
        t.add( std::to_string(Ms[j]) );
    }

    t.endOfRow();

    for (size_t i = 0; i < Ns.size(); ++i)
    {
        t.add( std::to_string(Ns[i]) );

        for (size_t j = 0; j < Ms.size(); ++j)
        {
            t.add( std::to_string(values[i][j]) );
        }

        t.endOfRow();
    }

    std::cout << t;
}

int main()
{
    table();
    // testSystemOfLinearEquationsSolver();
    
    return 0;
}