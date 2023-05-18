#include <iostream>
#include <functional>
#include <vector>
#include <cmath>
#include <string>
#include <vector>
#define _OPENMP_LLVM_RUNTIME
#include <omp.h>
#define FMT_HEADER_ONLY
#include <fmt/core.h>

#include "text_table.h"

double riemann_sum_double_integral(std::function<double(double, double)> f, double a, double b, double c, double d, int nx, int ny) {
    double hx = (b - a) / nx;
    double hy = (d - c) / ny;
    double integral = 0.0;

#pragma omp parallel for collapse(2) schedule(runtime) reduction (+:integral)
    for (int i = 0; i < nx; ++i)
    {
        for (int j = 0; j < ny; ++j)
        {
            double x = a + (i + 0.5) * hx;
            double y = c + (j + 0.5) * hy;

            integral += f(x, y) * hx * hy;
        }
    }

    return integral;
}

void measurement() {
    std::vector<int> N{ 10, 100, 1000, 10000 };
    std::vector<int> M{ 2, 4, 8, 16 };

    double a = 0.0;
    double b = 1.0;
    double c = 0.0;
    double d = 1.0;

    double A = 1.0;
    double B = 2.0;
    double C = 2.0;

    std::function<std::function<double(double, double)>(double, double, double)> F = [](double A, double B, double C) {
        return [A, B, C](double x, double y) {
            return A * (std::pow(x, B) + std::pow(y, C));
        };
    };

    // f(x, y) = A * (x^B + y^C)
    std::function<double(double, double)> f = F(A, B, C);

    std::vector<double> seq;

    for (size_t i = 0; i < N.size(); ++i)
    {
        omp_set_num_threads(1);
        omp_set_schedule(omp_sched_static, 0);

        double start = omp_get_wtime();
        double I = riemann_sum_double_integral(f, a, b, c, d, N[i], N[i]);
        double end = omp_get_wtime();

        double elapsed = end - start;

        std::cout << "Elapsed time: " << elapsed << "s." << std::endl;

        seq.push_back(elapsed);
    }

    std::vector<std::vector<double>> mul(N.size());

    for (int i = 0; i < N.size(); ++i)
    {
        for (int j = 0; j < M.size(); ++j)
        {
            omp_set_num_threads(M[j]);
            omp_set_schedule(omp_sched_static, 0);

            double start = omp_get_wtime();
            double I = riemann_sum_double_integral(f, a, b, c, d, N[i], N[i]);
            double end = omp_get_wtime();

            double elapsed = end - start;

            std::cout << "Elapsed time: " << elapsed << "s." << std::endl;

            mul[i].push_back(elapsed);
        }
    }

    TextTable t( '-', '|', '+' );

    t.add( "N" );
    t.add( "M" );
    t.add( "T_1" );
    t.add( "T_p" );
    t.add( "S_p" );
    t.add( "E_p" );
    t.endOfRow();

    for (int i = 0; i < N.size(); ++i)
    {
        for (int j = 0; j < M.size(); ++j)
        {
            t.add( std::to_string( N[i] ) );
            t.add( std::to_string( M[j] ) );
            t.add( std::to_string( seq[i] ) );
            t.add( std::to_string( mul[i][j] ) );
            t.add( std::to_string( seq[i] / mul[i][j] ) );
            t.add( std::to_string( seq[i] / (M[j] * mul[i][j]) ) );
            t.endOfRow();
        }
    }

    std::cout << t;
}

int main(void) {
    measurement();
}