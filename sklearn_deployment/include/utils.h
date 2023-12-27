#pragma once

#include <codecvt>
#include <fstream>
#include <vector>


namespace utils
{
	size_t vectorProduct(const std::vector<int64_t>& vector);
	std::wstring charToWstring(const char* str);
}
