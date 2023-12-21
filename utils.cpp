#include "utils.h"

#include <exception>
#include <string>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>

using Eigen::Matrix;
using Eigen::Vector;
using Eigen::Dynamic;


int load_csv(std::string fpath, 
            std::vector<Vector<double, Dynamic>*>& x_data){ 
    std::ifstream fin;
    fin.open(fpath);
    std::string line, word; 
    std::vector<double> temp;
    temp.reserve(1000);

    size_t lines{0};
    try{
        while(fin){
            std::getline(fin, line); 
            std::stringstream s(line);

            size_t l{0};
            while(std::getline(s, word, ',')){
                temp.push_back(std::stod(word));
                l += 1;
            }

            Vector<double, Dynamic>* v = new Vector<double, Dynamic>(l);
            for(int i{0}; i < l; i++){
                (*v)[i] = temp[i];
            }
            x_data.push_back(v);
            temp.clear();

            if(lines > 20)
                break;
            lines+=1;
        }
    } catch(const std::exception& exception){
            fin.close();
            return 0;
    }
    std::cout << "Samples: " << lines << '\n';
    fin.close();
    return 1;
}

