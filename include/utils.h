#ifndef UTILS_H
#define UTILS_H

#include <exception>
#include <string>
#include <string_view>
#include <iostream>
#include <fstream>
#include <vector>
#include "typedefs.h"
#include "pngWrapper.h"

template<typename ArgType>
inline int load_csv(std::string fpath, ArgType& data){
    typedef typename ArgType::Scalar Scalar;

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
    data = TensorMap<ArgType> (temp.data(), 
           temp.size()/lines, lines);
    std::cout << "Size: " << temp.size() << "\n\n";
    // Each column is a sample
    return 1;
}

inline int load_csv(std::string fpath, Tensor<std::string, 2>& data)
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
    data = TensorMap<Tensor<std::string, 2>> (temp.data(), 
           temp.size()/lines, lines);
    std::cout << "Size: " << temp.size() << "\n\n";
    // Each column is a sample
    return 1;
}

template<typename ArgType>
inline void imwrite(ArgType im, std::string path) {
    static_assert(ArgType::NumDimensions == 2);
    typedef typename Eigen::internal::traits<ArgType>::Index TensorIndex;
    typedef typename Eigen::internal::traits<ArgType>::Scalar ImScalar;

    TensorRef<Tensor<ImScalar, Eigen::internal::traits<ArgType>::NumDimensions,
                    Eigen::internal::traits<ArgType>::Layout, TensorIndex>>
        im_ref(im);
    
    Index height = im_ref.dimension(0);
	Index width = im_ref.dimension(1);

    Tensor<byte, 2> im_norm = im.unaryExpr(max_normalize_op(im, 255)).cast<byte>();
    std::ofstream fpo(path + ".png", std::ios::binary);
    png::PNGWriter writer(fpo, 
            png::pngInfo(height, width));
    writer.write(PNG_COLOR_TYPE_GRAY, im_norm);
}

#if 0
inline void imwrite(Tensor<float, 2>& im, std::string path){

    std::ofstream fpo(path + ".png", std::ios::binary);
    png::PNGWriter writer(fpo, 
            png::pngInfo(im.dimension(0), im.dimension(1)));

    Tensor<byte, 2> im_norm = 
        im.unaryExpr(max_normalize_op(im, 255)).cast<byte>();
    writer.write(PNG_COLOR_TYPE_GRAY, im_norm);
}

inline void imwrite(Tensor<float, 2>&& im, std::string&& path){
    imwrite(im, path);
}
#endif

#endif