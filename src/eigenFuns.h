#ifndef FUNCTORS_H
#define FUNCTORS_H

#include <functional>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::array;
using Eigen::Tensor;

template<typename T>
struct scalar_comapre_op{
    scalar_comapre_op(std::function<T(T, T)> condition, const T& v0) 
    :m_condition{condition}, m_v0{v0}{std::cout << "v0: " << m_v0 << " ";}

    void reduce(const T t, T* accum) const{
        *accum = m_condition(t, *accum);
    }
    template <typename Packet>
    void reducePacket(const Packet& packet, Packet* acc) const {
        (*acc) = Eigen::internal::padd<Packet>(*acc, packet);
    }

    T initialize() const{
        return T(m_v0);
    }
    template <typename Packet>
    Packet initializePacket() const {
        float init = initialize();
        return Eigen::internal::pset1<Packet>(init);
    }

    T finalize(const T accum) const{
        return accum;
    }
    template <typename Packet>
    Packet finalizePacket(const Packet& acc) const {
        return acc;
    }
    template <typename Packet>
    T finalizeBoth(const T acc_val, const Packet& acc_packet) const {
        auto packet = Eigen::internal::predux(acc_packet);
        return this->custom_op(acc_val, packet);
    }
private:
    std::function<T(T, T)> m_condition;
    const T& m_v0;
};

template<typename T>
T min(const Tensor<T, 2>& t)
{
    array<Eigen::Index, 2> dims({0, 1});
    scalar_comapre_op<T> comparer([](T x, T y)->T {return x < y ? x: y;}, t(0, 0));
    Tensor<T, 0> temp = t.reduce(dims, comparer);
    return temp(0);
}

template<typename T>
T max(const Tensor<T, 2>& t)
{
    array<Eigen::Index, 2> dims({0, 1});
    scalar_comapre_op<T> comparer( [](T x, T y)->T {return x > y ? x: y;}, t(0, 0));
    Tensor<T, 0> temp = t.reduce(dims, comparer);
    std::cout << "temp " << temp << " " << temp(0) << "\n";
    return temp(0);
}

inline const Eigen::array<int, 2> shuffle_transpose({1, 0});
template<typename ArgType>
ArgType
transposed(const ArgType& t){
    return t.shuffle(shuffle_transpose);
}

struct max_normalize_op
{
    max_normalize_op(Tensor<float, 2> t)
        :m_max{max<float>(t)}, m_min{min<float>(t)}
    {
        std::cout << "minmax: " << m_max << ", " << m_min << "\n";
    }
    float operator()(float a) const{
        return (a- m_min) / (m_max-m_min);
    }
private:
    float m_max;
    float m_min;
};



#endif