#include "utils.h"

#include <exception>
#include <string>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>

using Eigen::Matrix;
using Eigen::Vector;
using Eigen::Map;
using Eigen::Dynamic;

int load_csv(std::string fpath, Matrix<double, Dynamic, Dynamic>& data){
    std::ifstream fin;
    fin.open(fpath);
    std::string line, word; 
    std::vector<double> temp;
    temp.reserve(1000);

    size_t lines{0};
    try{
        do{

            std::getline(fin, line); 
            std::stringstream s(line);

            while(std::getline(s, word, ',')){
                temp.push_back(std::stod(word));
            }

            lines+=1;
            if(lines > 1000)
                break;
        }while(fin);
    } catch(const std::exception& exception){
            fin.close();
            return 0;
    }

    //lines -= 1;
    std::cout << "Samples: " << lines << '\n';
    fin.close();
    data = Map<Matrix<double, Dynamic, Dynamic>>(temp.data(), 
           lines, temp.size()/lines);
    // Each column is a sample
    data.transposeInPlace();
    return 1;
}

int load_csv(std::string fpath, Matrix<float, Dynamic, Dynamic>& data){
    std::ifstream fin;
    fin.open(fpath);
    std::string line, word; 
    std::vector<float> temp;
    temp.reserve(1000);

    size_t lines{0};
    try{
        do{

            std::getline(fin, line); 
            std::stringstream s(line);

            while(std::getline(s, word, ',')){
                temp.push_back(std::stof(word));
            }

            lines+=1;
            //if(lines > 1000)
            //    break;
        }while(fin);
    } catch(const std::exception& exception){
            fin.close();
            return 0;
    }

    lines -= 1;
    std::cout << "Samples: " << lines << '\n';
    fin.close();
    data = Map<Matrix<float, Dynamic, Dynamic>>(temp.data(), 
           temp.size()/lines, lines);
    std::cout << "Size: " << temp.size() << "\n\n";
    // Each column is a sample
    return 1;
}