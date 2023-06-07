#include <iostream>
#define FMT_HEADER_ONLY
#include <fmt/core.h>
#include "text_table.h"
#include "helper.h"

#include <cmath>
#include <vector>
#include <chrono>

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


        for (size_t i = 0; i < N.size(); ++i) {
            std::string filename = fmt::format("N={}_M=1.txt", N[i]);
            std::string cmd = fmt::format("mpiexec -n 1 ./multi N {} FILENAME {}", N[i], filename);

            Helper::exec(cmd);

            std::string text;
            Helper::readFile(filename, text);

            double elapsed = std::stod(text);
            if (verbose) {
                std::cout << "Elapsed time: " << elapsed << "s." << std::endl;
            }

            Helper::deleteFile(filename);

            m_time.seq.set(i, elapsed);
        }

        std::vector<TimeFrame> mul;

        for (int i = 0; i < N.size(); ++i) {
            for (int j = 0; j < M.size(); ++j) {
                std::string filename = fmt::format("N={}_M={}.txt", N[i], M[j]);
                std::string cmd = fmt::format("mpiexec -n {} ./multi N {} FILENAME {}", M[j], N[i], filename);

                Helper::exec(cmd);

                std::string text;
                Helper::readFile(filename, text);

                double elapsed = std::stod(text);
                if (verbose) {
                    std::cout << "Elapsed time: " << elapsed << "s." << std::endl;
                }

                Helper::deleteFile(filename);

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


int main(int argc, char **argv) 
{
    Helper::exec("mpic++ multi.cpp -I include/ -o multi");

    std::vector<int> steps { 5000, 7500, 10000, 12500 };
    std::vector<int> threads { 2, 3, 4, 5 };

    SetupRecord setup = { steps, threads };
    Measurement measurement(setup);

    measurement.execute_mean(1, true);

    print_table(setup, measurement.time());

    return 0;
}