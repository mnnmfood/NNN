#ifndef PNG_WRAPPER_H
#define PNG_WRAPPER_H

#include <exception>
#include <string_view>
#include <string>
#include <vector>
#include <cstddef>

#include <unsupported/Eigen/CXX11/Tensor>
#include "typedefs.h"

#include <png.h>

using Eigen::Map;

namespace png{
class PNG_io_exception: public std::exception
{
    std::string m_error{};
public:
    PNG_io_exception(std::string& error)
        :m_error{error}
    {
    }
    const char* what() const noexcept override{return m_error.c_str();}
};

typedef struct pngInfo: png_info
{
    png_structp m_png;
    png_infop m_info_ptr;

    pngInfo(){}

    pngInfo(int _height, int _width, int _color_type){
        height = static_cast<png_uint_32>(_height);
        width = static_cast<png_uint_32>(_width);
        bit_depth = 8;
        color_type = _color_type;
        interlace_type = PNG_INTERLACE_NONE;
        compression_type = PNG_COMPRESSION_TYPE_DEFAULT;
        filter_type = PNG_FILTER_TYPE_DEFAULT;
        gamma = 0;
    }

    pngInfo(png_structp png)
        :m_png{png},
        m_info_ptr{png_create_info_struct(m_png)}
    {
        *static_cast<png_info*>(this) = *m_info_ptr;
    }

    void set_info_ptr(png_structp png, png_infop info){
        m_info_ptr = info;
        m_png = png;
        sync();
    }

    void read(){
        png_read_info(m_png, m_info_ptr);
        png_get_IHDR(m_png,
                    m_info_ptr,
                    &width,
                    &height,
                    reinterpret_cast<int*>(&bit_depth),
                    reinterpret_cast<int*>(&color_type),
                    reinterpret_cast<int*>(&interlace_type),
                    reinterpret_cast<int*>(&compression_type),
                    reinterpret_cast<int*>(&filter_type ));
    }

    void write(){
        png_write_info(m_png, m_info_ptr);
    }

    void update(){
        png_read_update_info(m_png, m_info_ptr);
        png_get_IHDR(m_png,
                    m_info_ptr,
                    &width,
                    &height,
                    reinterpret_cast<int*>(&bit_depth),
                    reinterpret_cast<int*>(&color_type),
                    reinterpret_cast<int*>(&interlace_type),
                    reinterpret_cast<int*>(&compression_type),
                    reinterpret_cast<int*>(&filter_type ));
    }

protected:
    void sync(){
        png_set_IHDR(m_png,
                    m_info_ptr,
                    width,
                    height,
                    bit_depth,
                    color_type,
                    interlace_type,
                    compression_type,
                    filter_type);
    }
} pngInfo;

class PNGReader
{
public:
    png_structp m_png;
    pngInfo m_info;

    std::string error_msg;
    size_t total_size;

    PNGReader() = delete;
    PNGReader(std::istream&);
    ~PNGReader();

    void read(int, std::vector<byte>&);
    void read(int, Tensor<byte, 2>&);
    void readRow(byte*);

    friend std::ostream& operator<<(std::ostream&, PNGReader&);

protected:
    std::vector<byte> read_vec(int);
    void setTransforms(int);
    static void read_callback(png_structp, byte*, png_size_t);
    static void raise_error(png_structp, char const*);
    void throw_error();
};

class PNGWriter
{
public:
    std::vector<byte> m_data;
    png_structp m_png;
    pngInfo m_info;

    std::string error_msg;

    PNGWriter() = delete;
    PNGWriter(std::ostream&, std::vector<byte>, pngInfo);
    PNGWriter(std::ostream&, Tensor<byte, 2>, pngInfo);
    ~PNGWriter();

    void write(int);
    void writeRow(byte*);

    friend std::ostream& operator<<(std::ostream&, PNGWriter&);

protected:
    void setTransforms(int);
    static void write_data(png_structp, byte*, png_size_t);
    static void flush_data(png_structp);
    static void raise_error(png_structp, char const*);
    void throw_error();
};



}//png

#endif