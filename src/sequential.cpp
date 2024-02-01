#include <chrono>
#include "sequential.h"


// Sequential model
Sequential2::Sequential2(std::initializer_list<BaseLayer*> layers, CostFun* cost)
    :_layers{layers}, _cost{cost}, num_layers{layers.size()}
{
    // connect forward
    _layers[0]->_prev = nullptr;
    BaseLayer* prev_layer = _layers[0];

    for(size_t i{1}; i < num_layers; i++){
        _layers[i]->_prev = prev_layer;
        _layers[i]->initParams();

        prev_layer = _layers[i];
    }
    // connect backwards
    BaseLayer* next_layer = nullptr;
    for(size_t i{num_layers}; i > 0; i--){
        _layers[i-1]->_next = next_layer;
        next_layer = _layers[i-1];
    }
}

Sequential2::~Sequential2(){
    delete _cost;
    for(size_t i{0}; i < num_layers; i++){
        delete _layers[i]; 
    }
}

void Sequential2::init(size_t num_samples){
    for(size_t i{0}; i < num_layers; i++){
        _layers[i]->init(num_samples);
    }
}

void Sequential2::fwdProp(Tensor<float, 2>& input){
    BaseLayer* layer = _layers.front();
    layer->fwd(TensorWrapper(input));
    layer = layer->next();
    while(layer){
        layer->fwd();
        layer = layer->next();
    }
}
void Sequential2::fwdProp(Tensor<float, 2>&& input){fwdProp(input);}

void Sequential2::bkwProp(Tensor<float, 2>& output){
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
void Sequential2::bkwProp(Tensor<float, 2>&& output){bkwProp(output);}

void Sequential2::SGD(Tensor<float, 2>& x,
                    Tensor<float, 2>& y, 
                    int epochs, int batch_size, float lr, float mu,
                    Tensor<float, 2>& val_x,
                    Tensor<float, 2>&val_y){
    size_t train_size = x.dimension(1);

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
            fwdProp(sliced(x, sub_indices, 1));
            bkwProp(sliced(y, sub_indices, 1));


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

float Sequential2::accuracy(Tensor<float, 2>& x, Tensor<float, 2>& y)
{
    Eigen::Index test_size{x.dimension(1)};
    init(test_size);
    fwdProp(x);
    std::array<Index, 1> shape = _layers.back()->
                    out_shape().get<std::array<Index, 1>>();
    std::array<Index, 2> batch_shape {shape[0], test_size};
    Tensor<float, 2> pred = _layers.back()
        ->get_act().get(batch_shape);

    Tensor<Eigen::Index, 0> y_pred;
    int sum{0};

    for(Eigen::Index i{0}; i < test_size; i++){
        //pred.col(i).maxCoeff(&y_pred);
        y_pred = pred.chip(i, 1).argmax();
        sum += (static_cast<int>(y(i)) == y_pred(0));
    }

    return static_cast<float>(sum) / static_cast<float>(test_size);
}