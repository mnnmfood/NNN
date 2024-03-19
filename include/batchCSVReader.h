
#ifndef BATCH_READER_CSV2
#define BATCH_READER_CSV2

#include <filesystem>
#include <random>
#include <exception>
#include "typedefs.h"
#include "batchReader.h"

#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/tokenizer.hpp>

namespace ip = boost::interprocess;
namespace io = boost::iostreams;
using std::filesystem::file_size;

struct BatchCSVIterator
{
	typedef typename traits<BatchCSVReader>::data_t it_t;
	typedef typename traits<BatchCSVReader>::out_data_t out_data_t;
	typedef typename traits<BatchCSVReader>::out_label_t out_label_t;
	typedef boost::tokenizer< boost::escaped_list_separator<char> , 
		std::string::const_iterator, std::string> Tokenizer;

	BatchCSVIterator(it_t* begin, Index batch, char* data_file, char* label_file, 
					Index num_data, Index num_labels)
		:_begin{ begin }, _batch{ batch }, _data_file(data_file), _label_file(label_file),
		_seps('\\', ',', '\"'), _num_data{ num_data }, _num_labels {num_labels}
	{
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
			read_line(_labels.data() + i * _num_labels, _label_file, 
				it->second.first, it->second.second);
		}
		return _labels;
	}

	out_data_t& data() {
		for (Index i{ 0 }; i < _batch; i++) {
			it_t* it = _begin + i;
			read_line(_data.data() + i * _num_data, _data_file, 
				it->first.first, it->first.second);
		}
		return _data;
	}

private:
	void read_line(float* data, char* ifs, Index off, Index size) {
		std::string s(size, '0');
		std::copy_n(ifs + off, size, s.data());
		Tokenizer tok(s, _seps);
		int i = 0;
		for (auto it : tok) {
			data[i] = static_cast<float>(std::stof(it));
			i++;
		}
	}
	


	it_t* _begin;
	Index _batch;
	char* _data_file;
	char* _label_file;
	Tensor<float, 2> _data;
	Tensor<float, 2> _labels;
	Index _num_labels;
	Index _num_data;
	boost::escaped_list_separator<char> _seps;
};

class BatchCSVReader: public BatchReader<BatchCSVReader>
{
	typedef io::stream_buffer<io::array_source> readonly_buffer;
	size_t _dsize;
	size_t _lsize;
	ip::file_mapping _data_file;
	ip::file_mapping _label_file;
	ip::mapped_region _data_reg;
	ip::mapped_region _label_reg;
	Index _num_data{ 0 };
	Index _num_labels{ 0 };
public: 

	BatchCSVReader(std::string data_file, std::string label_file, Index batch) :BatchReader(batch), 
		_dsize{file_size(data_file)}, _lsize{file_size(label_file)},
		_data_file(data_file.data(), ip::read_only), _label_file(label_file.data(), ip::read_only),
		_data_reg(_data_file, ip::read_only, 0, _dsize), 
		_label_reg(_label_file, ip::read_only, 0, _lsize)
	{
		readonly_buffer data_buff((char*)(_data_reg.get_address()), _data_reg.get_size());
		readonly_buffer label_buff((char*)(_label_reg.get_address()), _label_reg.get_size());
		std::istream data_stream(&data_buff);
		std::istream label_stream(&label_buff);

		std::string s;
		Index off1 = 0, off2=0;
		while (data_stream && label_stream) {
			std::getline(data_stream, s);
			std::getline(label_stream, s);
			if (s.length() > 0) {
				_path_arr.emplace_back(
					std::make_pair(
						std::make_pair(off1, data_stream.tellg() - off1),
						std::make_pair(off2, label_stream.tellg() - off2)
						)
				);
				off1 = data_stream.tellg();
				off2 = label_stream.tellg();
			}
		}
		_data = _path_arr.data();
		_total_size = _path_arr.size();

		label_stream.clear();
		data_stream.clear();
		label_stream.seekg(0);
		data_stream.seekg(0);
		_num_data = peek_size(data_stream);
		_num_labels = peek_size(label_stream);
	}

	static Index peek_size(std::istream& ifs) {
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
	BatchCSVIterator iter(data_t* data, Index batch) { 
		return BatchCSVIterator(data, batch, 
			(char*) _data_reg.get_address(), 
			(char*) _label_reg.get_address(), 
			_num_data, _num_labels); 
	}
};

#endif
