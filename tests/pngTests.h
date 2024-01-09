
#include "pngWrapper.h"

void testColor()
{

    std::cout << "-Read Color" << "\n";
    // Read Image
    std::string dataDir {"../data/png/"};
    std::ifstream fpi{dataDir + "cs-black-000.png", std::ios::binary};
    png::PNGReader reader(fpi);
    Tensor<byte, 2> data;
    reader.read(PNG_COLOR_TYPE_RGB, data);
    std::cout << reader;

    std::cout << "-Write Color" << "\n";
    // Write Image
    std::string outDir {"./"};
    std::ofstream fpo(outDir + "cs-black-000_rgb.png", std::ios::binary);
    png::PNGWriter writer(fpo, png::pngInfo(reader.m_info.height, 
                    reader.m_info.width));
    writer.write(PNG_COLOR_TYPE_RGB, data);
    std::cout << writer;
}

void testGray()
{

    std::cout << "-Read Grayscale" << "\n";
    // Read Image
    std::string dataDir {"../data/png/"};
    std::ifstream fpi{dataDir + "cs-black-000.png", std::ios::binary};
    png::PNGReader reader(fpi);
    //Tensor<byte, 2> data;
    std::vector<byte> data;
    reader.read(PNG_COLOR_TYPE_GRAY, data);
    std::cout << reader;

    std::cout << "-Write Grayscale" << "\n";
    // Write Image
    std::string outDir {"./"};
    std::ofstream fpo(outDir + "cs-black-000_gray.png", std::ios::binary);
    png::PNGWriter writer(fpo, png::pngInfo(reader.m_info.height, 
                    reader.m_info.width));
    writer.write(PNG_COLOR_TYPE_GRAY, data);
    std::cout << writer;
}

void testReset()
{
    // First read-write
    std::string dataDir {"../data/png/"};
    std::ifstream fpi1{dataDir + "cs-black-000.png", std::ios::binary};
    png::PNGReader reader(fpi1);
    std::vector<byte> data;
    reader.read(PNG_COLOR_TYPE_GRAY, data);

    std::string outDir {"./"};
    std::ofstream fpo1(outDir + "out1.png", std::ios::binary);
    png::PNGWriter writer(fpo1, png::pngInfo(reader.m_info.height, 
                    reader.m_info.width));
    writer.write(PNG_COLOR_TYPE_GRAY, data);

    // Second read-write
    std::ifstream fpi2{dataDir + "cs-black-000.png", std::ios::binary};
    reader.reset(fpi2);
    reader.read(PNG_COLOR_TYPE_GRAY, data);

    std::ofstream fpo2(outDir + "out2.png", std::ios::binary);
    writer.reset(fpo2);
    writer.write(PNG_COLOR_TYPE_GRAY, data);
}

void testPNG()
{
    testColor();
    testGray();
    std::cout << "Success" << "\n\n";
}