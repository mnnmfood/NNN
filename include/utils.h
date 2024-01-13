#ifndef UTILS_H
#define UTILS_H

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

template<typename Scalar>
int load_csv(std::string fpath, Matrix<Scalar, Dynamic, Dynamic>& data){
    std::ifstream fin;
    fin.open(fpath);
    std::string line, word; 
    std::vector<Scalar> temp;
    temp.reserve(1000);

    size_t lines{0};
    try{
        do{

            std::getline(fin, line); 
            std::stringstream s(line);

            while(std::getline(s, word, ',')){
                temp.push_back(static_cast<Scalar>(std::stof(word)));
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
    data = Map<Matrix<Scalar, Dynamic, Dynamic>>(temp.data(), 
           temp.size()/lines, lines);
    std::cout << "Size: " << temp.size() << "\n\n";
    // Each column is a sample
    return 1;
}

int load_csv(std::string fpath, Matrix<std::string, Dynamic, Dynamic>&data)
{
    std::ifstream fin;
    fin.open(fpath);
    std::string line, word; 
    std::vector<std::string> temp;
    temp.reserve(1000);

    size_t lines{0};
    try{
        do{

            std::getline(fin, line); 
            std::stringstream s(line);

            while(std::getline(s, word, ',')){
                temp.push_back(static_cast<std::string>(word));
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
    data = Map<Matrix<std::string, Dynamic, Dynamic>>(temp.data(), 
           temp.size()/lines, lines);
    std::cout << "Size: " << temp.size() << "\n\n";
    // Each column is a sample
    return 1;
}



#endif