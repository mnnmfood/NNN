#ifndef FUNCTORS_H
#define FUNCTORS_H

#include <functional>
#include "typedefs.h"

// ---- Sum Matrix and Vector colwise or rowwise

struct VecsumOp
{
  VecsumOp(Tensor<float, 2>& mat, Tensor<float, 1>& vec, bool rowwise=true)
    :m_mat{mat}, m_vec{vec}, m_rows{mat.dimension(0)}, m_cols{mat.dimension(1)}, 
    m_rowwise{rowwise}
  {
    if(rowwise){
      assert(m_mat.dimension(1)==m_vec.dimension(0));
    }else{
      assert(m_mat.dimension(0)==m_vec.dimension(0));
    }
  } 
  const float operator()(Eigen::Index idx) const{
    Eigen::Index row = idx % m_rows; 
    Eigen::Index col = idx / m_rows;
    return m_mat(row, col) + (m_rowwise?m_vec(col):m_vec(row));
  }
private:
  Tensor<float, 2> m_mat;
  Tensor<float, 1> m_vec;
  Index m_rows;
  Index m_cols;
  bool m_rowwise;
};

// L-value version
inline auto vecSum(Tensor<float, 2>& mat, Tensor<float, 1>& vec, bool rowwise=true){
    return mat.nullaryExpr(VecsumOp(mat, vec, rowwise));
}

// R-value version
inline auto vecSum(Tensor<float, 2>&& mat, Tensor<float, 1>& vec, bool rowwise=true){
    return mat.nullaryExpr(VecsumOp(mat, vec, rowwise));
}
// ---

// --- Reducer: calculate squared norm of either rows or cols
template <typename T> 
struct SqNormReducer
{
   EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) const {
     Eigen::internal::scalar_sum_op<T> sum_op;
     *accum = sum_op(*accum, t);
   }
   template <typename Packet>
   EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reducePacket(const Packet& p, Packet* accum) const {
     (*accum) = Eigen::internal::padd<Packet>(*accum, p);
   }
  
   EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
     Eigen::internal::scalar_cast_op<int, T> conv;
     return conv(0);
   }
   template <typename Packet>
   EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet initializePacket() const {
     return Eigen::internal::pset1<Packet>(initialize());
   }
   EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T accum) const {
     Eigen::internal::scalar_sqrt_op<T> sqrt_op;
     return sqrt_op(accum);
   }
   template <typename Packet>
   EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet finalizePacket(const Packet& vaccum) const {
     return vaccum;
   }
   template <typename Packet>
   EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalizeBoth(const T saccum, const Packet& vaccum) const {
     Eigen::internal::scalar_sum_op<T> sum_op;
     return sum_op(saccum, predux(vaccum));
   }
 };

// --- Reducer: compare elements by condition
template<typename T>
struct scalar_comparer_op{
    scalar_comparer_op(std::function<T(T, T)> condition)
    :m_condition{condition}, is_first{true} {}

    void reduce(const T t, T* accum){
        if(is_first){
            *accum = t;
            is_first = false;
        }else{
            *accum = m_condition(t, *accum);
        }
    }
    template <typename Packet>
    void reducePacket(const Packet& packet, Packet* acc) const {
        (*acc) = Eigen::internal::padd<Packet>(*acc, packet);
    }

    T initialize() const{
        return T(0);
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
    bool is_first;
};

// --- Reducer: cacluate min/max value
template<typename ArgType>
typename ArgType::Scalar min(const ArgType& t)
{
    using Scalar = typename ArgType::Scalar;
    std::array<Index, ArgType::NumDimensions> dims;
    for(size_t i{0}; i < ArgType::NumDimensions; i++){
      dims[i] = static_cast<Index>(i);
    }
    scalar_comparer_op<Scalar> comparer([](Scalar x, Scalar y)->Scalar {return x < y ? x: y;});
    Eigen::Tensor<Scalar, 0> temp = t.reduce(dims, comparer);
    return temp(0);
}

template<typename ArgType>
typename ArgType::Scalar max(const ArgType& t)
{
    using Scalar = typename ArgType::Scalar;
    std::array<Index, ArgType::NumDimensions> dims;
    for(size_t i{0}; i < ArgType::NumDimensions; i++){
      dims[i] = static_cast<Index>(i);
    }
    scalar_comparer_op<Scalar> comparer([](Scalar x, Scalar y)->Scalar {return x > y ? x: y;});
    Eigen::Tensor<Scalar, 0> temp = t.reduce(dims, comparer);
    return temp(0);
}
// ---

// --- Transpose tensor: by its first 2 dimensions
template<typename ArgType>
auto
transposed(const ArgType& t){
    std::array<int, ArgType::NumDimensions> shuffle_transpose;
    shuffle_transpose[0] = 1;
    shuffle_transpose[1] = 0;
    for(int i{2}; i < ArgType::NumDimensions; i++)
        shuffle_transpose[i] = i;
    return t.shuffle(shuffle_transpose);
}

// --- Unary-Op: normalize tensor to range 0-1
template<typename ArgType>
struct max_normalize_op
{
    using Scalar = typename ArgType::Scalar;
    max_normalize_op(ArgType t, int mult)
        :m_max{max<ArgType>(t)}, m_min{min<ArgType>(t)},
        m_mult{mult}
    {
    }
    Scalar operator()(Scalar a) const{
        return m_mult * (a - m_min) / (m_max-m_min);
    }
private:
    Scalar m_max;
    Scalar m_min;
    int m_mult;
};

// -- Slice tensor along given dimension:
template<typename ArgType>
auto sliced(const ArgType& arg, const std::vector<int>& indices, int dim)
{
  std::array<Eigen::Index, ArgType::NumDimensions> out_size;
  for(int i{0}; i < ArgType::NumDimensions; i++){
    out_size[i] = arg.dimension(i);
  }
  out_size[dim] = indices.size();

  ArgType out_slice(out_size);
  for(size_t i{0}; i < indices.size(); i++){
    out_slice.chip(i, dim) = arg.chip(indices[i], dim);
  }
  return out_slice;
}

template<typename ArgType>
auto sliced(const ArgType&& arg, const std::vector<int>& indices, int dim)
{
  assert(indices.size() < arg.dimension(dim));
  assert(*std::max_element(indices.begin(), indices.end()) < arg.dimension(dim));
  std::array<Eigen::Index, ArgType::NumDimensions> out_size;
  for(int i{0}; i < ArgType::NumDimensions; i++){
    out_size[i] = arg.dimension(i);
  }
  out_size[dim] = indices.size();

  ArgType out_slice(out_size);
  for(size_t i{0}; i < indices.size(); i++){
    out_slice.chip(i, dim) = arg.chip(indices[i], dim);
  }
  return out_slice;
}

#endif