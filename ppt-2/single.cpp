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


int main(void) {
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

    omp_set_num_threads(1);

    double start = omp_get_wtime();
    double I = riemann_sum_double_integral(f, a, b, c, d, 100, 100);
    double end = omp_get_wtime();

    double elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed << "s." << std::endl;

    std::cout << fmt::format(
        "integrate ({}) dx from x={} to {} = {}",
        fmt::format("integrate ({} * (x^{} + y^{})) dx from x={} to {}", A, B, C, c, d),
        a, b, I
    ) << std::endl;
}