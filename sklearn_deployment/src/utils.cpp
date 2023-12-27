#include "utils.h"

size_t utils::vectorProduct(const std::vector<int64_t>& vector)
{
    if (vector.empty())
        return 0;

    size_t product = 1;
    for (const auto& element : vector)
        product *= element;
    return product;
}

std::wstring utils::charToWstring(const char* str)
{
    typedef std::codecvt_utf8<wchar_t> convert_type;
    std::wstring_convert<convert_type, wchar_t> converter;
    return converter.from_bytes(str);  // codecvt_utf8来将UTF-8编码的std::string转换为UTF-16编码的std::wstring。
}








