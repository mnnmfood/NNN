#include <iostream>
#include "pngWrapper.h"
#include "eigenFunctors.h"

namespace png
{
PNGReader::PNGReader(std::istream& stream)
    :m_png{png_create_read_struct(PNG_LIBPNG_VER_STRING, 
        this, raise_error, 0)},
    m_info {m_png}
{
    png_set_read_fn(m_png, &stream, read_callback);
    m_info.read();
}

void PNGReader::read(int dst_type, std::vector<byte>& data)
{
    data = read_vec(dst_type);
}

void PNGReader::read(int dst_type, Tensor<byte, 2>& data)
{
    // there is porbably a better way to do it, for now
    // return tranpsosed tensor
    Tensor<byte, 2> temp = TensorMap<Tensor<byte, 2>>(
        read_vec(dst_type).data(),
        m_info.width,
        m_info.height
    );
    data = transposed(temp);
}

std::vector<byte> PNGReader::read_vec(int dst_type)
{
    setTransforms(dst_type);
    int nbytes = png_get_rowbytes(m_png, m_info.m_info_ptr);

    total_size = nbytes * m_info.height;
    std::vector<byte> data;
    data.reserve(total_size);
    data.resize(total_size);
    for(png_size_t i{0}; i < m_info.height; i++){
        readRow(&data[nbytes*i]);
    }
    return data;
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
        m_info.color_type == PNG_COLOR_TYPE_RGB_ALPHA) 
        && (dst_color == PNG_COLOR_TYPE_GRAY)){
        png_set_rgb_to_gray(m_png, 1, -1, -1);
    }
    // If dst is rgb
    else if((m_info.color_type == PNG_COLOR_TYPE_GRAY ||
        m_info.color_type == PNG_COLOR_TYPE_GRAY_ALPHA) 
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

PNGReader::~PNGReader()
{
    png_destroy_read_struct(&m_png, &m_info.m_info_ptr, NULL);
}

} // end png