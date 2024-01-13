#include <iostream>
#include "pngWrapper.h"

namespace png
{
PNGReader::PNGReader(std::ifstream& stream)
    :m_png{png_create_read_struct(PNG_LIBPNG_VER_STRING, 
        this, raise_error, 0)},
    m_info {m_png}, m_stream {&stream}
{
    png_set_read_fn(m_png, m_stream, read_callback);
    m_info.read();
}

void PNGReader::read(int dst_type, std::vector<byte>& data)
{
    setTransforms(dst_type);
    nbytes = png_get_rowbytes(m_png, m_info.m_info_ptr);
    data.resize(nbytes * m_info.height);
    read_arr(data.data(), m_info.height, nbytes, dst_type);
}

void PNGReader::read(int dst_type, Tensor<byte, 2>& data)
{
    if(dst_type != m_info.color_type)
        setTransforms(dst_type);

    nbytes = png_get_rowbytes(m_png, m_info.m_info_ptr);
    // there is porbably a better way to do it, for now
    // return tranpsosed tensor
    data = Tensor<byte, 2>(nbytes, m_info.height);
    read_arr(data.data(), m_info.height, nbytes, dst_type);
    data = transposed(data);
}

void PNGReader::read_arr(byte* buffer, size_t rows, size_t cols, int dst_type)
{
    // at this point buffer memory should be allocated
    for(size_t i{0}; i < rows; i++){
        readRow(buffer + cols*i);
    }
}

void PNGReader::setTransforms(int dst_color)
{
    // Reduce bit depth to 8
    if(m_info.bit_depth == 16)
        png_set_strip_16(m_png);
    // Or augment 
    if(m_info.bit_depth < 8)
        png_set_packing(m_png);
    // Remove alpha channel
    if(m_info.color_type & PNG_COLOR_MASK_ALPHA)
        png_set_strip_alpha(m_png);
    // If palette set to rgb
    if(m_info.color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(m_png);

    if((dst_color != PNG_COLOR_TYPE_GRAY) &&
        (dst_color != PNG_COLOR_TYPE_RGB)){
        error_msg = "Color Type not supported";
        throw_error();
    }
    // If dst is gray-scale
    else if((m_info.color_type == PNG_COLOR_TYPE_RGB ||
        m_info.color_type == PNG_COLOR_TYPE_RGB_ALPHA ||
        m_info.color_type == PNG_COLOR_TYPE_PALETTE) 
        && (dst_color == PNG_COLOR_TYPE_GRAY)){
        png_set_rgb_to_gray(m_png, 1, -1, -1);
    }
    // If dst is rgb
    else if((m_info.color_type == PNG_COLOR_TYPE_GRAY ||
        m_info.color_type == PNG_COLOR_TYPE_GRAY_ALPHA ||
        m_info.color_type == PNG_COLOR_TYPE_PALETTE) 
        && (dst_color == PNG_COLOR_TYPE_RGB)){
        png_set_gray_to_rgb(m_png);
    }
    m_info.update();
}

void PNGReader::readRow(byte* bytes)
{
    png_read_row(m_png, bytes, 0);
}

void PNGReader::read_callback(png_structp png, byte* data, png_size_t length)
{
    PNGReader* image = static_cast<PNGReader*>(png_get_error_ptr(png));
    std::istream* stream = reinterpret_cast<std::istream*>(png_get_io_ptr(png));
    try
    {
        stream->read(reinterpret_cast<char*>(data), length);
    }
    catch(const std::exception& e)
    {
        image->error_msg = "IO exception";
        image->throw_error();
    }
    catch(...)
    {
    }
}

void PNGReader::throw_error()
{
    throw PNG_io_exception(error_msg);
}

void PNGReader::raise_error(png_structp png, char const* message)
{
    PNGReader* reader = static_cast<PNGReader*>(png_get_error_ptr(png));
    reader->error_msg = message;
    reader->throw_error();
}

std::ostream& operator<<(std::ostream& stream, PNGReader& reader){
    stream << "\n";
    stream << "--- Color: ";
    if(reader.m_info.color_type == PNG_COLOR_TYPE_GRAY)
        stream << "Gray";
    if(reader.m_info.color_type == PNG_COLOR_TYPE_PALETTE)
        stream << "Palette";
    if(reader.m_info.color_type == PNG_COLOR_TYPE_RGB)
        stream << "RGB";
    if(reader.m_info.color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        stream << "Gray-alpha";
    if(reader.m_info.color_type == PNG_COLOR_TYPE_RGBA)
        stream << "RGB-alpha";
    stream << "\nChannel Depth: ";
    stream << static_cast<int>(reader.m_info.bit_depth);
    stream << "\n";
    stream << "Shape: " << static_cast<int>(reader.m_info.height) << ", ";
    stream << static_cast<int>(reader.m_info.width) << "\n\n";
    return stream;

}

void PNGReader::reset(std::ifstream& stream)
{
    m_stream = &stream;
    // There is no way around it, png structs cannot be reused
    png_destroy_read_struct(&m_png, &m_info.m_info_ptr, NULL);
    m_png = png_create_read_struct(PNG_LIBPNG_VER_STRING, 
        this, raise_error, 0);
    m_info = pngInfo(m_png);
    
    png_set_read_fn(m_png, m_stream, read_callback);
    m_info.read();
}

PNGReader::~PNGReader()
{
    png_destroy_read_struct(&m_png, &m_info.m_info_ptr, NULL);
}

} // end png