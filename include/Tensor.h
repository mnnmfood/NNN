#ifndef TENSOR_H
#define TENSOR_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>

using Eigen::Index;

template<typename T, int NumDimensions>
using TensorView = Eigen::TensorMap<Eigen::Tensor<T, NumDimensions>>;

template<typename T>
class Tensor
{
    template<size_t NumDimensions>
    bool checkSize(std::array<Index, NumDimensions>& size){
        size_t total_size = 1;
        for(size_t n{0}; n < NumDimensions; n++){
            total_size *= size[n];
        }
        return _size == total_size;
    }
public:
    T* data = nullptr;
    size_t _size;

    Tensor(){}

    Tensor(size_t size)
        :_size{size}
    {
        data = new T[_size];
    }

    Tensor(const Tensor<T>& t){
        _size = t._size;
        data = new T[_size];
        memcpy(data, t.data, sizeof(T)*_size);
    }

    Tensor<T>& operator=(const Tensor<T>& t){
        _size = t._size;
        if(data)
            delete[] data;
        data = new T[_size];
        memcpy(data, t.data, sizeof(T)*_size);
        return *this;
    }

    Tensor<T>& operator=(const Tensor<T>&& t){
        _size = t._size;
        if(data)
            delete[] data;
        data = new T[_size];
        memcpy(data, t.data, sizeof(T)*_size);
        return *this;
    }
    // Return as Eigen 
    template<size_t NumDimensions>
    Eigen::TensorMap<Eigen::Tensor<T, NumDimensions>> tensor(std::array<Index, NumDimensions>& size){
        assert(checkSize(size));
        return Eigen::TensorMap<Eigen::Tensor<T, NumDimensions>>(data.get(), size);
    }

    Eigen::TensorMap<Eigen::Tensor<T, 1>> tensor(){
        return Eigen::TensorMap<Eigen::Tensor<T, 1>>(data, _size);
    }

    template<typename Ts>
    friend std::ostream& operator<<(std::ostream& out, const Tensor<Ts>& tensor);

    ~Tensor(){
        if(data)
            delete[] data;
    }
};

template<typename T>
std::ostream& operator<<(std::ostream& out, const Tensor<T>& tensor){
    for(size_t i{0}; i < tensor._size; i++){
        out << tensor.data[i] << ", ";
    }
    out << "\n\n";
    return out;
}

#endif