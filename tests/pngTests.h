#include <sys/stat.h>
#include "pngWrapper.h"
#include "layers.h"

inline const std::string data_dir {"../data/"};
inline const std::string out_dir {"./"};

void testColor()
{

    std::cout << "-Read Color" << "\n";
    // Read Image
    std::string dataDir {data_dir + "png/"};
    std::ifstream fpi{dataDir + "cs-black-000.png", std::ios::binary};
    png::PNGReader reader(fpi);
    Tensor<byte, 2> data;
    reader.read(PNG_COLOR_TYPE_RGB, data);
    std::cout << reader;

    std::cout << "-Write Color" << "\n";
    // Write Image
    std::ofstream fpo(out_dir + "cs-black-000_rgb.png", std::ios::binary);
    std::cout << reader.m_info.width << "\n";
    png::PNGWriter writer(fpo, png::pngInfo(reader.m_info.height, 
                    reader.m_info.width));
    writer.write(PNG_COLOR_TYPE_RGB, data);
    std::cout << writer;
}

void testGray()
{

    std::cout << "-Read Grayscale" << "\n";
    // Read Image
    std::string dataDir {data_dir + "png/"};
    std::ifstream fpi{dataDir + "cs-black-000.png", std::ios::binary};
    png::PNGReader reader(fpi);
    std::vector<byte> data;
    reader.read(PNG_COLOR_TYPE_GRAY, data);
    std::cout << reader;

    std::cout << "-Write Grayscale" << "\n";
    // Write Image
    std::ofstream fpo(out_dir + "cs-black-000_gray.png", std::ios::binary);
    png::PNGWriter writer(fpo, png::pngInfo(reader.m_info.height, 
                    reader.m_info.width));
    writer.write(PNG_COLOR_TYPE_GRAY, data);
    std::cout << writer;
}

void testReset()
{
    // First read-write
    std::string dataDir {data_dir + "png/"};
    std::ifstream fpi1{dataDir + "cs-black-000.png", std::ios::binary};
    png::PNGReader reader(fpi1);
    std::vector<byte> data;
    reader.read(PNG_COLOR_TYPE_GRAY, data);

    std::ofstream fpo1(out_dir + "basn0g01.png", std::ios::binary);
    png::PNGWriter writer(fpo1, png::pngInfo(reader.m_info.height, 
                    reader.m_info.width));
    writer.write(PNG_COLOR_TYPE_GRAY, data);

    // Second read-write
    std::ifstream fpi2{dataDir + "cs-black-000.png", std::ios::binary};
    reader.reset(fpi2);
    reader.read(PNG_COLOR_TYPE_GRAY, data);

    std::ofstream fpo2(out_dir + "basn0g02.png", std::ios::binary);
    writer.reset(fpo2, png::pngInfo(reader.m_info.height, reader.m_info.width));
    writer.write(PNG_COLOR_TYPE_GRAY, data);
}

void testBulk()
{
    std::cout << "-- Test Bulk" << "\n";
    int batch_size = 32;
    std::string fullpath {data_dir + "mnist_png/testing/0/"};
    Tensor<std::string, 2> index;
    load_csv(fullpath + "index.csv", index);

    std::ifstream fp{fullpath + index(0, 0), std::ios::binary};
    png::PNGReader reader{fp};
    size_t width = reader.m_info.width;
    size_t height = reader.m_info.height;
    size_t total = width * height;
    std::cout << width << ", " << height << "\n";

    Tensor<byte, 3> images(height, width, batch_size);
    for(int i{0}; i < batch_size; i++){
        std::ifstream fpi(fullpath + index(0, i), std::ios::binary);
        reader.reset(fpi);
        reader.read_arr(images.data() + total * i, width, height, PNG_COLOR_TYPE_GRAY);
    }
    images = transposed(images);
    for(int i{0}; i < batch_size; i++){
        Tensor<byte, 2> image0 = images.chip(i, 2);
        std::ofstream fpo(out_dir + "bulk_" + std::to_string(i) + ".png", std::ios::binary);
        png::PNGWriter writer(fpo, png::pngInfo(width, height));
        //writer.write_arr(image0.data(), width, height, PNG_COLOR_TYPE_GRAY);
        writer.write(PNG_COLOR_TYPE_GRAY, image0);
    }
}

void testConvolve(){
    std::cout << "-- Test Convolution" << "\n";
    std::string dataDir {"../data/mnist_csv/"};
    Tensor<float, 2> data;
    load_csv(dataDir + "train_x.csv", data);
    Tensor<float, 2> x;
    x = data.slice(std::array<Index, 2>({0, 0}), 
        std::array<Index, 2>({data.dimension(0), 10}));
    Sequential2 model ({
        new ReshapeLayer<1, 3>(std::array<Index, 3>{28, 28, 1}),
        new ConvolLayer(std::array<Index, 3>{3, 3, 1})}, 
        new MSE(),
        std::array<Index, 1> {784}, std::array<Index, 3>{26, 26, 1}
        );
    size_t batch_size = x.dimension(1); 
    model.init(batch_size);
    model.fwdProp(x);
    Tensor<float, 4> result = model.output(batch_size);

    for(size_t i{0}; i < batch_size; i++){
        Tensor<float, 2> image = result.chip(i, 3).chip(0, 2);
        Tensor<byte, 2> image_norm = 
            image.unaryExpr(max_normalize_op(result, 255)).cast<byte>();

        std::ofstream fpo(out_dir + "convol" + std::to_string(i) + ".png", std::ios::binary);
        png::PNGWriter writer(fpo, 
                png::pngInfo(image.dimension(0), image.dimension(1)));
        writer.write(PNG_COLOR_TYPE_GRAY, image_norm);
    }
}

void testPNG()
{
    testConvolve();
    testColor();
    testGray();
    testReset();

    struct stat info;
    std::string pathname {data_dir + "mnist_png"};
    if(stat(pathname.data(), &info)==0)
        testBulk();
    std::cout << "Success" << "\n\n";
}