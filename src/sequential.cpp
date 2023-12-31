#include <chrono>
#include "layers.h"
#include "sequential.h"


// Sequential model
Sequential2::Sequential2(std::initializer_list<Layer*> layers, CostFun* cost)
    :_layers{layers}, _cost{cost}, num_layers{layers.size()}
{

    // connect forward
    _layers[0]->_prev = nullptr;
    Layer* prev_layer = _layers[0];
    size_t prev_size = _layers[0]->_size;

    for(size_t i{1}; i < num_layers; i++){
        _layers[i]->_prev = prev_layer;
        _layers[i]->initParams(prev_size);

        prev_size = _layers[i]->_size;
        prev_layer = _layers[i];
    }
    // connect backwards
    Layer* next_layer = nullptr;
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

void Sequential2::fwdProp(const MatrixXf& input){
    Layer* layer = _layers.front();
    layer->fwd(input);
    layer = layer->next();
    while(layer){
        layer->fwd();
        layer = layer->next();
    }
}

void Sequential2::bkwProp(const MatrixXf& output){
    Layer* layer = _layers.back();
    layer->bwd(
        _cost->grad(layer->get_act(), output)
        );
    layer = layer->prev();
    while(layer){
        layer->bwd();
        layer = layer->prev();
    }
}

void Sequential2::SGD(Matrix<float, Dynamic, Dynamic>& x,
                    Matrix<float, Dynamic, Dynamic>& y, 
                    int epochs, int batch_size, float lr, float mu,
                    Matrix<float, Dynamic, Dynamic>& val_x,
                    Matrix<float, Dynamic, Dynamic>&val_y){
    size_t train_size = x.cols();
    init(batch_size);

    // Prepare random indices 
    std::vector<int> indices;
    std::vector<int> sub_indices(batch_size);
    for(size_t i{0}; i < train_size; i++){indices.push_back(i);} 

    for(int k{0}; k < epochs; k++){

        auto start = std::chrono::high_resolution_clock::now();
        std::shuffle(indices.begin(), indices.end(), gen);

        for(size_t l{0}; l < train_size-batch_size; l+=batch_size){

            std::copy_n(indices.begin()+l, batch_size, sub_indices.begin());
            fwdProp(x(all, sub_indices)); 
            bkwProp(y(all, sub_indices));


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

float Sequential2::accuracy(const MatrixXf& x, const MatrixXf& y)
{
    Eigen::Index test_size{x.cols()};

    fwdProp(x);
    MatrixXf pred = _layers.back()->get_act();

    int y_pred;
    int sum{0};

    for(Eigen::Index i{0}; i < test_size; i++){
        pred.col(i).maxCoeff(&y_pred);
        sum += (static_cast<int>(y(i)) == y_pred);
    }

    return static_cast<float>(sum) / static_cast<float>(test_size);
}