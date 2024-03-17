
#ifndef BATCH_READER_CSV
#define BATCH_READER_CSV

#include <filesystem>
#include <random>
#include <exception>
#include "typedefs.h"
#include "utils.h"
#include "batchReader.h"

struct BatchCSVIterator
{
	typedef typename traits<BatchCSVReader>::data_t it_t;
	typedef typename traits<BatchCSVReader>::out_data_t out_data_t;
	typedef typename traits<BatchCSVReader>::out_label_t out_label_t;

	BatchCSVIterator(it_t* begin, Index batch, std::ifstream& data_file, std::ifstream& label_file)
		:_begin{ begin }, _batch{ batch }, _data_file(data_file), _label_file(label_file)
	{
		_num_labels = peek_size(_label_file);
		_num_data = peek_size(_data_file);
		_labels = Tensor<float, 2>(_num_labels, _batch);
		_data = Tensor<float, 2>(_num_data, _batch);
	}

	BatchCSVIterator& operator ++() {
		_begin += _batch;
		return *this;
	}

	BatchCSVIterator operator ++(int) {
		BatchCSVIterator temp = *this;
		++(*this);
		return temp;
	}

	BatchCSVIterator& operator --() {
		_begin -= _batch;
		return *this;
	}

	BatchCSVIterator operator --(int) {
		BatchCSVIterator temp = *this;
		--(*this);
		return temp;
	}
	
	friend bool operator==(BatchCSVIterator& a, BatchCSVIterator& b) { return a._begin == b._begin; }
	friend bool operator!=(BatchCSVIterator& a, BatchCSVIterator& b) { return a._begin != b._begin; }
	
	out_label_t& labels() {
		for (Index i{ 0 }; i < _batch; i++) {
			it_t* it = _begin + i;
			read_line(_labels.data() + i * _num_labels, _label_file, it->first);
		}
		return _labels;
	}

	out_data_t& data() {
		for (Index i{ 0 }; i < _batch; i++) {
			it_t* it = _begin + i;
			read_line(_data.data() + i * _num_data, _data_file, it->second);
		}
		return _data;
	}

private:
	void read_line(float* data, std::ifstream& ifs, Index off) {
		std::string word, line;
		ifs.seekg(off);
		std::getline(ifs, line);
		std::stringstream s(line);
		int i = 0;
		while (std::getline(s, word, ',')) {
			data[i] = static_cast<float>(std::stof(word));
			i++;
		}
	}
	
	Index peek_size(std::ifstream& ifs) {
		//read first line to get data size
		std::string word, line;
		std::getline(ifs, line);
		std::stringstream s(line);
		Index i = 0;
		while (std::getline(s, word, ',')) {
			i++;
		}
		return i;
	}

	it_t* _begin;
	Index _batch;
	std::ifstream& _data_file;
	std::ifstream& _label_file;
	Tensor<float, 2> _data;
	Tensor<float, 2> _labels;
	Index _num_labels;
	Index _num_data;
};

class BatchCSVReader: public BatchReader<BatchCSVReader>
{
	std::ifstream _data_file;
	std::ifstream _label_file;

public: 

	BatchCSVReader(std::string data_file, std::string label_file, Index batch)
		:BatchReader(batch), _data_file(data_file), _label_file(label_file)
	{
		std::string s;
		Index off1 = 0, off2=0;
		while (_data_file) {
			std::getline(_label_file, s);
			std::getline(_data_file, s);
			if (s.length() > 0) {
				_path_arr.emplace_back(
					std::make_pair(off1, off2)
				);
				_total_size++;
				off1 = _label_file.tellg();
				off2 = _data_file.tellg();
			}
		}
		_label_file = std::ifstream(label_file);
		_data_file = std::ifstream(data_file);
		_data = _path_arr.data();
		_total_size = _path_arr.size();
	}
	
	BatchCSVIterator iter(data_t* data, Index batch) { return BatchCSVIterator(data, batch, _data_file, _label_file); }
};

#endif
