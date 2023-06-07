#ifndef HELPER_H
#define HELPER_H

#include <vector>
#include <algorithm>
#include <random>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

#include <sys/stat.h>
#include <unistd.h>
#include <fstream>

class Helper
{
public:
	static void set_seed(int seed)
	{
		s_seed = seed;
	}

	static std::vector<double> random(int size, int low, int high)
	{
		std::vector<double> result(size);

		std::mt19937 gen(s_seed);
		std::uniform_int_distribution<int> dis(low, high);

		for (int i = 0; i < result.size(); ++i) {
			result[i] = dis(gen);
		}

		return result;
	}

	static std::string exec(const std::string &cmd) 
    {
        const char *converted_cmd = cmd.c_str();
        std::array<char, 128> buffer;
        std::string result;
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(converted_cmd, "r"), pclose);
        if (!pipe) {
            throw std::runtime_error("popen() failed!");
        }
        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        }
        return result;
    }

    static inline bool exists(const std::string &name) 
    {
        return (access( name.c_str(), F_OK ) != -1);
    }


    static void readFile(const std::string &filename, std::string &text) 
    {
        std::ifstream file(filename);
        while (std::getline(file, text)) {}

        file.close();
    }

    static void writeFile(const std::string &filename, const std::string &text)
    {
        std::ofstream file(filename);
        file << text;
        file.close();
    }

    static void deleteFile(const std::string &filename)
    {
        std::remove(filename.c_str());
    }

private:
	static int s_seed;
};

int Helper::s_seed = 3;

#endif // HELPER_H