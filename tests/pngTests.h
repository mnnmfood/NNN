
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
    png::PNGWriter writer(fpo, data, png::pngInfo(reader.m_info.height, 
                    reader.m_info.width));
    writer.write(PNG_COLOR_TYPE_RGB);
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
    png::PNGWriter writer(fpo, data, png::pngInfo(reader.m_info.height, 
                    reader.m_info.width));
    writer.write(PNG_COLOR_TYPE_GRAY);
    std::cout << writer;
}

void testPNG()
{
    testColor();
    testGray();
    std::cout << "Success" << "\n\n";
}