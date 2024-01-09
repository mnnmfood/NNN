#include <iostream>
#include "pngWrapper.h"

namespace png
{
PNGWriter::PNGWriter(std::ostream& stream, 
                    pngInfo info)
    :m_info{info}
{
    m_png  = png_create_write_struct(
        PNG_LIBPNG_VER_STRING, 
        this, 
        raise_error, 
        0);
    m_info.set_info_ptr(m_png, png_create_info_struct(m_png));
    png_set_write_fn(m_png, &stream, write_data, flush_data);
}

void PNGWriter::write(int dst_type, std::vector<byte>& data)
{
    nbytes = data.size() / m_info.height;
    write_arr(data.data(), m_info.height, nbytes, dst_type);
}

void PNGWriter::write(int dst_type, Tensor<byte, 2> data)
{
    // In eigen tensor data is ordered column-wise so we
    // need to transpose
    data = transposed(data);

    nbytes = data.size() / m_info.height;
    write_arr(data.data(), m_info.height, nbytes, dst_type);
}

void PNGWriter::write_arr(byte* buffer, size_t rows, size_t cols, int dst_type)
{
    setTransforms(dst_type);
    m_info.write();

    for(size_t i{0}; i < rows; i++){
        writeRow(buffer + cols*i);
    }
}

void PNGWriter::setTransforms(int dst_color)
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

void PNGWriter::writeRow(byte* bytes)
{
    png_write_row(m_png, bytes);
}

void PNGWriter::write_data(png_structp png, byte* data, png_size_t length)
{
    PNGWriter* writer = static_cast<PNGWriter*>(png_get_error_ptr(png));
    std::ostream* stream = reinterpret_cast<std::ostream*>(png_get_io_ptr(png));
    try
    {
        stream->write(reinterpret_cast<char*>(data), length);
        if(!stream->good()){
            writer->error_msg = "Error in stream write";
            writer->throw_error();
        }
    }
    catch(const std::exception& e)
    {
        writer->error_msg = "IO exception";
        writer->throw_error();
    }
    catch(...)
    {
    }
}

void PNGWriter::flush_data(png_structp png)
{
    PNGWriter* writer = static_cast<PNGWriter*>(png_get_error_ptr(png));
    std::ostream* stream = reinterpret_cast<std::ostream*>(png_get_io_ptr(png));
    try
    {
        stream->flush();
        if(!stream->good()){
            writer->error_msg = "Error in stream write";
            writer->throw_error();
        }
    }
    catch(const std::exception& e)
    {
        writer->error_msg = "IO exception";
        writer->throw_error();
    }
}

void PNGWriter::throw_error()
{
    throw PNG_io_exception(error_msg);
}

void PNGWriter::raise_error(png_structp png, char const* message)
{
    PNGWriter* reader = static_cast<PNGWriter*>(png_get_error_ptr(png));
    reader->error_msg = message;
    reader->throw_error();
}

std::ostream& operator<<(std::ostream& stream, PNGWriter& reader){
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

PNGWriter::~PNGWriter()
{
    png_destroy_write_struct(&m_png, NULL);
}

} //end png