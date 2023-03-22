#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

class CSVReader
{
	static vector<vector<double>> read_csv(string filename) {
        vector<vector<double>> data;
        ifstream file(filename);

        string line;
        while (getline(file, line)) {
            vector<double> row;
            stringstream ss(line);

            string cell;
            while (getline(ss, cell, ',')) {
                row.push_back(stod(cell));
            }

            data.push_back(row);
        }

        return data;
    }
};