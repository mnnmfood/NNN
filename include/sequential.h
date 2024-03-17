#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include <vector>
#include <initializer_list>
#include "typedefs.h"
#include "batchPNGReader.h"
#include "layers.h"
#include "costs.h"
#include "timer.h"

template<size_t num_dims_in, size_t num_dims_out>
class Sequential2
{
    typedef Tensor<float, num_dims_in + 1> in_batch_t;
    typedef Tensor<float, num_dims_out + 1> out_batch_t;

    std::vector<BaseLayer*> _layers;
    CostFun* _cost;
    const size_t num_layers;
    std::array<Index, num_dims_in> _in_shape;
    std::array<Index, num_dims_out> _out_shape;
    ThreadPool* _pool;
    Eigen::ThreadPoolDevice* _device;
public:
    Sequential2(std::initializer_list<BaseLayer*> layers, std::array<Index, num_dims_in> in_shape, 
        std::array<Index, num_dims_out> out_shape, CostFun* cost = new DummyCost<num_dims_out>())
    :_layers{layers}, _cost{cost}, num_layers{_layers.size() + 2},
    _in_shape{in_shape}, _out_shape{out_shape}{
        
        // Initialize device
        const int pool_n{ 8 };
        const int thread_n{ 4 };
        this->_pool = new ThreadPool(pool_n);
        this->_device = new ThreadPoolDevice(_pool, thread_n);

        // Add input and output layers
        _layers.insert(_layers.begin(), new InputLayer(_in_shape));
        _layers.push_back(new OutputLayer(_out_shape, cost));

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
    void bkwProp(out_batch_t& output){
        BaseLayer* layer = _layers.back();
        layer->bwd(TensorWrapper(output), _device);
        //layer->bwd(TensorWrapper(
        //    _cost->grad((layer->get_act()).get(output.dimensions()), output)
        //    ),
        //    _device
        //);
        layer = layer->prev();

        while(layer){
            layer->bwd(_device);
            layer = layer->prev();
        }
    }
    void fwdProp(in_batch_t& input){
        BaseLayer* layer = _layers.front();
        layer->fwd(TensorWrapper(input), _device);
        layer = layer->next();
        while(layer){
            layer->fwd(_device);
            layer = layer->next();
        }
    }
    void bkwProp(out_batch_t&& output){bkwProp(output);}
    void fwdProp(in_batch_t&& input){fwdProp(input);}
   
    template<class reader>
    void SGD(reader& train_reader, int epochs, float lr,
        float mu, reader& val_reader) {
        typedef reader::out_data_t data_t;
        // check if read data type matches input data type
        static_assert(std::is_same<in_batch_t, data_t>::value);

        Timer timer;
        for (int k{ 0 }; k < epochs; k++) {
            init(train_reader.batch());
            train_reader.reset();
            timer.start();
            auto end = train_reader.end();
            for (auto it = train_reader.begin(); it != end; it++) {
                fwdProp(it.data());
                bkwProp(it.labels());
                for (size_t i{ 0 }; i < num_layers; i++) {
                    _layers[i]->update(lr, mu, train_reader.batch());
                }
            }
            timer.stop();
            std::cout << "Epoch " << k + 1 << "\n";
            float cost_t = accuracy(val_reader);
            std::cout << "Accuracy: " << cost_t * 100 << " %" << "\n";
            std::cout << "Time: " << timer.elapsedMilliseconds() << "ms\n";
        }
    }
#if 0 
    void SGD(BatchPNGReader train_reader, int epochs, float lr,
        float mu, BatchPNGReader val_reader){
        Timer timer;
        //Tensor<float, 2> labels;
        //Tensor<byte, 3> images;
        for(int k{0}; k < epochs; k++){
            init(train_reader.batch());
            train_reader.reset();
            timer.start();
            auto end = train_reader.end();
            for(auto it=train_reader.begin(); it!=end; it++){
                //train_reader.get(labels, images);
                fwdProp(it.images().cast<float>());
                bkwProp(it.labels());
                //fwdProp(images.cast<float>());
                //bkwProp(labels);
                for(size_t i{0}; i < num_layers; i++){
                    _layers[i]->update(lr, mu, train_reader.batch());                
                }
            }
			timer.stop();
            std::cout << "Epoch " << k + 1 << "\n";
			float cost_t = accuracy(val_reader);
			std::cout << "Accuracy: " << cost_t*100 << " %" << "\n";
            std::cout << "Time: " << timer.elapsedMilliseconds() << "ms\n";
        }

    }
#endif

    void SGD(in_batch_t& x, out_batch_t& y, int epochs, int batch_size, 
        float lr, float mu, in_batch_t& val_x, out_batch_t& val_y){
        Timer timer;
        size_t train_size = x.dimension(num_dims_in);

        // Prepare random indices 
        std::vector<int> indices(train_size);
        std::vector<int> sub_indices(batch_size);
        for(size_t i{0}; i < train_size; i++){indices.push_back(i);} 

        for(int k{0}; k < epochs; k++){
            init(batch_size);
            timer.start();
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
            timer.stop();
            std::cout << "Time: " << timer.elapsedMilliseconds() << "ms\n";
        }
    }
    
    template<class reader>
    float accuracy(reader& val_reader) {
        typedef reader::out_label_t label_t;
        const Eigen::Index test_size{val_reader.size()};
        const Eigen::Index batch_size{val_reader.batch()};
        val_reader.reset();
        int sum = 0;
		label_t labels;
        Tensor<Index, 0> y;
        Tensor<Index, 0> y_pred;
        auto end = val_reader.end();
        for(auto it = val_reader.begin(); it!=end;it++){
            init(batch_size);
            fwdProp(it.data());
            labels = it.labels();
            Tensor<float, 2> pred = _layers.back()
                ->get_act().get(labels.dimensions());
            for (Eigen::Index i{ 0 }; i < batch_size; i++) {
                y_pred = pred.chip(i, 1).argmax();
                y = labels.chip(i, 1).argmax();
                sum += (static_cast<int>(y(0)) == y_pred(0));
            }
        }
        return static_cast<float>(sum) / static_cast<float>(test_size);
    }

    float accuracy(in_batch_t& x, out_batch_t& y){
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
            y_pred = pred.chip(i, num_dims_out).argmax();
            sum += (static_cast<int>(y(i)) == y_pred(0));
        }

        return static_cast<float>(sum) / static_cast<float>(test_size);
    }
    float accuracy(in_batch_t&& x, out_batch_t&& y){
        return accuracy(x, y);
    }

    out_batch_t output(Index batch_size){
        std::array<Index, num_dims_out + 1> temp;
        std::copy(_out_shape.begin(), _out_shape.end(), 
            temp.begin());
        temp.back() = batch_size;
        return _layers.back()->get_act().get(temp);
    }

    ~Sequential2() {
        for (size_t i{ 0 }; i < num_layers; i++) {
            delete _layers[i];
        }
        delete _pool;
        delete _device;
    }
};

#endif