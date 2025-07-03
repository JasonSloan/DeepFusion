#include <dirent.h>
#include <fstream>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <dlnne/dlnne.h>

#include "opencv2/opencv.hpp"
#include "spdlog/logger.h"                 // spdlog日志相关
#include "spdlog/sinks/basic_file_sink.h"  // spdlog日志相关
#include "spdlog/spdlog.h"                 // spdlog日志相关
#include "yolo/model-utils.h"
#include "yolo/yolo.h"

using namespace std;
using namespace dl::nne;

template <typename _T>
shared_ptr<_T> make_nvshared(_T* ptr) {
    return shared_ptr<_T>(ptr, [](_T* p) { p->Destroy(); });
}

void compile_model(std::string modelPath_rlym, std::string modelPath_engine, int max_batch_size) {
    spdlog::info("Start to build model of path: {}", modelPath_rlym);

    BuilderConfig builder_config;
    builder_config.max_batch_size = max_batch_size;
    shared_ptr<Builder> builder = make_nvshared(CreateInferBuilder());
    builder->SetBuilderConfig(builder_config);
    shared_ptr<Network> network = make_nvshared(builder->CreateNetwork());
    shared_ptr<Parser> parser = make_nvshared(CreateParser());
    parser->RegisterInput("images", Dims4(1, 3, 640, 640), kNCHW);
    parser->RegisterOutput("output");
    parser->Parse(modelPath_rlym.c_str(), *network);
    shared_ptr<Engine> engine = make_nvshared(builder->BuildEngine(*network));
    shared_ptr<HostMemory> serializedModel = make_nvshared(engine->Serialize()); 
    save_file(modelPath_engine, serializedModel->Data(), serializedModel->Size());

    spdlog::info("Model built successfully, saved to {}", modelPath_engine);
}

void reIndexResults(vector<Result>& infer_results,
                    std::map<int, string>& src_map,
                    std::map<string, int>& tgt_map) {
    for (auto& result : infer_results)
        for (auto& bbox : result.rboxes) {
            string class_name = src_map[bbox.label];
            int tgt_label = tgt_map[class_name];
            bbox.label = tgt_label;
        }
}

std::map<int, std::string> readFileToMap(const std::string& filePath) {
    std::map<int, std::string> indexToClassMap;
    std::ifstream file(filePath);
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        int index;
        std::string className;
        lineStream >> index >> className;
        indexToClassMap[index] = className;
    }
    return indexToClassMap;
}

std::string getSuffix(const std::string& filePath) {
    size_t dotPosition = filePath.find_last_of('.');
    if (dotPosition == std::string::npos || dotPosition == 0) {
        return "";  // No extension found
    }
    return filePath.substr(dotPosition + 1);
}

std::string replaceSuffix(const std::string& path,
                          const std::string& oldSuffix,
                          const std::string& newSuffix) {
    std::string newPath = path;
    std::size_t pos = newPath.rfind(oldSuffix);
    if (pos != std::string::npos &&
        pos == newPath.length() - oldSuffix.length()) {
        newPath.replace(pos, oldSuffix.length(), newSuffix);
    }
    return newPath;
}

string get_logfile_name(string& log_dir) {
    if (access(log_dir.c_str(), 0) != F_OK)
        mkdir(log_dir.c_str(), S_IRWXU);
    DIR* pDir = opendir(log_dir.c_str());
    struct dirent* ptr;
    vector<string> files_vector;
    while ((ptr = readdir(pDir)) != nullptr) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            files_vector.push_back(ptr->d_name);
        }
    }
    closedir(pDir);
    std::sort(files_vector.begin(), files_vector.end());
    int max_num = 0;
    if (files_vector.size() != 0) {
        for (auto& file : files_vector) {
            string num_str = file.substr(0, file.find("."));
            int num = std::stoi(num_str);
            if (num > max_num)
                max_num = num;
        }
        max_num += 1;
    }
    return std::to_string(max_num);
}

bool save_file(const string& file, const void* data, size_t length) {
    FILE* f = fopen(file.c_str(), "wb");
    if (!f)
        return false;
    if (data && length > 0) {
        if (fwrite(data, 1, length, f) not_eq length) {
            fclose(f);
            return false;
        }
    }
    fclose(f);
    return true;
};

vector<unsigned char> load_file(const string& file) {
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0) {
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

bool endswith(const std::string& str, const std::string& ending) {
    if (str.length() >= ending.length()) {
        return str.compare(str.length() - ending.length(), ending.length(),
                           ending) == 0;
    }
    return false;
}

std::string getFileName(const std::string& file_path, bool with_ext=true){
	int index = file_path.find_last_of('/');
	if (index < 0)
		index = file_path.find_last_of('\\');
    std::string tmp = file_path.substr(index + 1);
    if (with_ext)
        return tmp;

    std::string img_name = tmp.substr(0, tmp.find_last_of('.'));
    return img_name;
}

std::vector<std::string> splitString(const std::string &str, const std::string &delim){
    std::vector<std::string> val;

    std::string::size_type pos1, pos2;
    pos2 = str.find(delim);
    pos1 = 0;
    while (std::string::npos != pos2)
    {
        val.push_back(str.substr(pos1, pos2 - pos1));

        pos1 = pos2 + delim.size();
        pos2 = str.find(delim, pos1);
    }
    if (pos1 != str.length())
        val.push_back(str.substr(pos1));
    return val;
}