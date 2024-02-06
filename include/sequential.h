#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include <Eigen/Dense>
#include <vector>
#include <initializer_list>
#include "typedefs.h"
#include "layers.h"
#include "costs.h"

template<size_t num_dims_in, size_t num_dims_out>
class Sequential2
{

    std::vector<BaseLayer*> _layers;
    CostFun* _cost;
    const size_t num_layers;
    std::array<Index, num_dims_in> _in_shape;
    std::array<Index, num_dims_out> _out_shape;
public:
    Sequential2(std::initializer_list<BaseLayer*> layers, CostFun* cost, 
        std::array<Index, num_dims_in> in_shape, std::array<Index, num_dims_out> out_shape)
    :_layers{layers}, _cost{cost}, num_layers{_layers.size() + 2},
    _in_shape{in_shape}, _out_shape{out_shape}{
        // Add input and output layers
        _layers.insert(_layers.begin(), new InputLayer(_in_shape));
        _layers.push_back(new OutputLayer(_out_shape));
        // connect forward
        _layers[0]->_prev = nullptr;
        BaseLayer* prev_layer = _layers[0];

        for(size_t i{1}; i < num_layers; i++){
            _layers[i]->_prev = prev_layer;
            _layers[i-1]->initParams();
            prev_layer = _layers[i];
        }
        // connect backwards
        BaseLayer* next_layer = nullptr;
        for(size_t i{num_layers}; i > 0; i--){
            _layers[i-1]->_next = next_layer;
            next_layer = _layers[i-1];
        }
     }
    void init(size_t batch_size){
        for(size_t i{0}; i < num_layers; i++){
            _layers[i]->init(batch_size);
        }
    }
    void bkwProp(Tensor<float, 2>& output){
        BaseLayer* layer = _layers.back();
        layer->bwd(TensorWrapper(
            _cost->grad((layer->get_act()).get(output.dimensions()), output)
        ));
        layer = layer->prev();
        while(layer){
            layer->bwd();
            layer = layer->prev();
        }
    }
    void fwdProp(Tensor<float, 2>& input){
        BaseLayer* layer = _layers.front();
        layer->fwd(TensorWrapper(input));
        layer = layer->next();
        while(layer){
            layer->fwd();
            layer = layer->next();
        }
    }
    void bkwProp(Tensor<float, 2>&& output){bkwProp(output);}
    void fwdProp(Tensor<float, 2>&& input){fwdProp(input);}

    void SGD(Tensor<float, num_dims_in+1>& x,
            Tensor<float, num_dims_out+1>& y, 
            int epochs, int batch_size, float lr, float mu,
            Tensor<float, num_dims_in+1>& val_x,
            Tensor<float, 2>&val_y){

        size_t train_size = x.dimension(num_dims_in);

        // Prepare random indices 
        std::vector<int> indices(train_size);
        std::vector<int> sub_indices(batch_size);
        for(size_t i{0}; i < train_size; i++){indices.push_back(i);} 

        for(int k{0}; k < epochs; k++){
            init(batch_size);
            auto start = std::chrono::high_resolution_clock::now();
            std::shuffle(indices.begin(), indices.end(), gen);

            for(size_t l{0}; l < train_size-batch_size; l+=batch_size){

                std::copy_n(indices.begin()+l, batch_size, sub_indices.begin());
                fwdProp(sliced(x, sub_indices, num_dims_in));
                bkwProp(sliced(y, sub_indices, num_dims_out));


                for(size_t i{0}; i < num_layers; i++){
                    _layers[i]->update(lr, mu, batch_size);                
                }
            }

            float cost_t = accuracy(val_x, val_y);
            std::cout << "Epoch " << k << " :" << cost_t*100; 
            std::cout << " %" << "\n";
            auto finish = std::chrono::high_resolution_clock::now();
            std::cout << "Time: " << 
                std::chrono::duration_cast<std::chrono::milliseconds>(finish-start).count() 
                << "ms\n";
        }
    }

    float accuracy(Tensor<float, num_dims_in + 1>& x, Tensor<float, 2>& y){
        Eigen::Index test_size{x.dimension(num_dims_in)};
        init(test_size);
        fwdProp(x);
        std::array<Index, num_dims_out + 1> batch_shape;
        for(size_t i{0}; i < num_dims_out; i++){
            batch_shape[i] = _out_shape[i];
        }
        batch_shape.back() = test_size;
        Tensor<float, 2> pred = _layers.back()
            ->get_act().get(batch_shape);

        Tensor<Eigen::Index, 0> y_pred;
        int sum{0};

        for(Eigen::Index i{0}; i < test_size; i++){
            //pred.col(i).maxCoeff(&y_pred);
            y_pred = pred.chip(i, num_dims_out).argmax();
            sum += (static_cast<int>(y(i)) == y_pred(0));
        }

        return static_cast<float>(sum) / static_cast<float>(test_size);
    }

    float accuracy(Tensor<float, num_dims_in>&& x, Tensor<float, 2>&& y){
        return accuracy(x, y);
    }

    Tensor<float, num_dims_out + 1> output(Index batch_size){
        std::array<Index, num_dims_out + 1> temp;
        std::copy(_out_shape.begin(), _out_shape.end(), 
            temp.begin());
        temp.back() = batch_size;
        return _layers.back()->get_act().get(temp);
    }
};

#endif