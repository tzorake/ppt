#include <iostream>
#include <functional>
#include <vector>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#define _OPENMP_LLVM_RUNTIME
#include <omp.h>
#define FMT_HEADER_ONLY
#include <fmt/core.h>

#include "text_table.h"

double riemann_sum_double_integral(std::function<double(double, double)> f, double a, double b, double c, double d, int nx, int ny) {
    double hx = (b - a) / nx;
    double hy = (d - c) / ny;
    double integral = 0.0;

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

    int n = 1000;

    std::function<std::function<double(double, double)>(double, double, double)> F = [](double A, double B, double C) {
        return [A, B, C](double x, double y) {
            return A * (std::pow(x, B) + std::pow(y, C));
        };
    };

    // f(x, y) = A * (x^B + y^C)
    std::function<double(double, double)> f = F(A, B, C);

    auto start = std::chrono::high_resolution_clock::now();
    double I = riemann_sum_double_integral(f, a, b, c, d, n, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    double elapsed = duration.count();

    std::cout << fmt::format(
        "integrate ({}) dx from x={} to {} = {}",
        fmt::format("integrate ({} * (x^{} + y^{})) dx from x={} to {}", A, B, C, c, d),
        a, b, I
    ) << std::endl;

    std::cout << fmt::format("Steps (the same for x and y): {}", n) << std::endl;
    std::cout << fmt::format("Elapsed time: {}s.", elapsed) << std::endl;
}