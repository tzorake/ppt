#ifndef UTILITIES_H
#define UTILITIES_H

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <fstream>

namespace FileSystem 
{
    std::string exec(const std::string &cmd) 
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

    inline bool exists(const std::string &name) 
    {
        return (access( name.c_str(), F_OK ) != -1);
    }


    void readFile(const std::string &filename, std::string &text) 
    {
        std::ifstream file(filename);
        while (std::getline(file, text)) {}

        file.close();
    }

    void writeFile(const std::string &filename, const std::string &text)
    {
        std::ofstream file(filename);
        file << text;
        file.close();
    }

    void deleteFile(const std::string &filename)
    {
        std::remove(filename.c_str());
    }
};

#endif
