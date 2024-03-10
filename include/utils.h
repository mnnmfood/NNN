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
inline int load_csv(std::string fpath, ArgType& data, int total_lines=-1){
    const bool not_full_file = total_lines > 0;
    typedef typename ArgType::Scalar Scalar;

    std::ifstream fin;
    fin.open(fpath);
    std::string line, word; 
    std::vector<Scalar> temp;
    temp.reserve(1000);

    size_t lines{0};
    try {
        while (fin){
            std::getline(fin, line);
            std::stringstream s(line);

            while (std::getline(s, word, ',')) {
                temp.push_back(static_cast<Scalar>(std::stof(word)));
            }
            lines++;

            if (not_full_file && (lines > (total_lines-1))) {
                lines++;
                break;
            }

        }
    } catch(const std::exception& exception){
            fin.close();
            return 0;
    }
    lines--;
    fin.close();
    data = TensorMap<ArgType> (temp.data(), 
           temp.size()/lines, lines);
    // Each column is a sample
    return 1;
}


inline int load_csv(std::string fpath, Tensor<std::string, 2>& data, int total_lines=-1)
{
    const bool not_full_file = total_lines > 0;

    std::ifstream fin;
    fin.open(fpath);
    std::string line, word; 
    std::vector<std::string> temp;
    temp.reserve(1000);

    size_t lines{0};
    try {
        while (fin){
            std::getline(fin, line);
            std::stringstream s(line);

            while (std::getline(s, word, ',')) {
                temp.push_back(static_cast<std::string>(word));
            }
            lines++;

            if (not_full_file && (lines > (total_lines-1))) {
                lines++;
                break;
            }

        }
    } catch(const std::exception& exception){
            fin.close();
            return 0;
    }
    lines--;
    fin.close();
    data = TensorMap<Tensor<std::string, 2>> (temp.data(), 
           temp.size()/lines, lines);
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

    Tensor<byte, 2> im_norm;
    if (typeid(ImScalar) != typeid(byte)) {
        Tensor<ImScalar, 2> temp = im.unaryExpr(max_normalize_op(im, 255));
        im_norm = temp.cast<byte>();
    }
    else {
        im_norm = im.cast<byte>();
    }
    std::ofstream fpo(path + ".png", std::ios::binary);
    png::PNGWriter writer(fpo, 
            png::pngInfo(height, width));
    writer.write(PNG_COLOR_TYPE_GRAY, im_norm);
}

#endif