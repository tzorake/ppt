#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <string>
#include <functional>
#define _OPENMP_LLVM_RUNTIME
#include <omp.h>
#include "text_table.h"

double riemann_sum_double_integral(std::function<double(double, double)> f, double a, double b, double c, double d, int nx, int ny) 
{
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

struct SetupRecord
{
    std::vector<int> steps;
    std::vector<int> threads;
};

struct TimeFrame
{
    std::vector<double> frame;

    void initialize(int n)
    {
        frame = std::move(std::vector<double>(n, 0.0));
    }

    void set(int i, double x)
    {
        frame[i] = x;
    }
};

struct TimeRecord 
{
    TimeFrame seq;
    std::vector<TimeFrame> mul;

    void initialize(SetupRecord setup)
    {
        int n_count = setup.steps.size();
        int m_count = setup.threads.size();

        seq.initialize(n_count);

        mul = std::vector<TimeFrame>(n_count);

        for (int i = 0; i < n_count; ++i)
        {
            mul[i].initialize(m_count);
        }
    }
};

void print_table(SetupRecord setup, TimeRecord time)
{
    std::vector<int> N = setup.steps;
    std::vector<int> M = setup.threads;
    TimeFrame seq = time.seq;
    std::vector<TimeFrame> mul = time.mul;

    TextTable table( '-', '|', '+' );

    table.add( "N" );
    table.add( "M" );
    table.add( "T_1" );
    table.add( "T_p" );
    table.add( "S_p" );
    table.add( "E_p" );
    table.endOfRow();

    for (int i = 0; i < N.size(); ++i)
    {
        for (int j = 0; j < M.size(); ++j)
        {
            int n = N[i];
            int m = M[j];

            double seq_time = seq.frame[i];
            double mul_time = mul[i].frame[j];

            table.add( std::to_string( n ));
            table.add( std::to_string( m ));
            table.add( std::to_string( seq_time ));
            table.add( std::to_string( mul_time ));
            table.add( std::to_string( seq_time / mul_time ));
            table.add( std::to_string( seq_time / (m * mul_time) ));
            table.endOfRow();
        }
    }

    std::cout << table << std::endl;
}

class Measurement
{
    SetupRecord m_setup;
    TimeRecord m_time;

public:
    Measurement(SetupRecord s): m_setup(s)
    {
        m_time.initialize(m_setup);
    }

    Measurement(Measurement const &other) = default;

    virtual ~Measurement() = default;
    
    Measurement &operator=(Measurement const &other) = default;

    Measurement operator+(Measurement const& rhs) const
    {
        Measurement result = *this;
        result += rhs;
        return result;
    }

    Measurement& operator+=(Measurement const& rhs)
    {
        auto &lhs_seq_data = m_time.seq.frame;
        auto &lhs_mul_vec = m_time.mul;

        auto &rhs_seq_data = rhs.m_time.seq.frame;
        auto &rhs_mul_vec = rhs.m_time.mul;

        for (int i = 0; i < lhs_seq_data.size(); ++i)
        {
            lhs_seq_data[i] += rhs_seq_data[i];
        }

        for (int i = 0; i < lhs_mul_vec.size(); ++i)
        {
            auto &lhs_mul_data = lhs_mul_vec[i].frame;
            auto &rhs_mul_data = rhs_mul_vec[i].frame;

            for (int j = 0; j < lhs_mul_data.size(); ++j)
            {
                lhs_mul_data[j] += rhs_mul_data[j];
            }
        }

        return *this;
    }

    Measurement operator/(int n) const
    {
        Measurement result = *this;
        result /= n;
        return result;
    }

    Measurement& operator/=(int n)
    {
        auto &lhs_seq_data = m_time.seq.frame;
        auto &lhs_mul_vec = m_time.mul;

        for (int i = 0; i < lhs_seq_data.size(); ++i)
        {
            lhs_seq_data[i] /= n;
        }

        for (int i = 0; i < lhs_mul_vec.size(); ++i)
        {
            auto &lhs_mul_data = lhs_mul_vec[i].frame;

            for (int j = 0; j < lhs_mul_data.size(); ++j)
            {
                lhs_mul_data[j] /= n;
            }
        }

        return *this;
    }

    void execute(bool verbose = false)
    {
        std::vector<int> N = m_setup.steps;
        std::vector<int> M = m_setup.threads;

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

        int n_count = N.size();
        int m_count = M.size();

        for (size_t i = 0; i < n_count; ++i)
        {
            omp_set_num_threads(1);
            omp_set_schedule(omp_sched_static, 0);

            double start = omp_get_wtime();
            double I = riemann_sum_double_integral(f, a, b, c, d, N[i], N[i]);
            double end = omp_get_wtime();

            double elapsed = end - start;

            if (verbose) {
                std::cout << "Elapsed time: " << elapsed << "s." << std::endl;
            }

            m_time.seq.set(i, elapsed);
        }

        std::vector<TimeFrame> mul;

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

                if (verbose) {
                    std::cout << "Elapsed time: " << elapsed << "s." << std::endl;
                }

                m_time.mul[i].set(j, elapsed);
            }
        }
    }

    TimeRecord time()
    {
        return m_time;
    }

    void execute_mean(int times, bool verbose = false)
    {
        m_time.initialize(m_setup);

        for (int i = 0; i < times; ++i) {
            Measurement measurement(m_setup);
            measurement.execute(verbose);

            (*this) += measurement;
        }

        (*this) /= times;
    }
};


int main(void) {
    std::vector<int> steps{ 1000, 2000, 5000, 10000 };
    std::vector<int> threads{ 2, 4, 8, 16 };

    SetupRecord setup = { steps, threads };
    Measurement measurement(setup);

    measurement.execute_mean(10, false);

    print_table(setup, measurement.time());
}