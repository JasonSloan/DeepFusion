#pragma once

#include <map>
#include <vector>
#include <chrono>
#include <string>

#include "yolo/yolo.h"

void compile_model(std::string modelPath_rlym, std::string modelPath_engine, int max_batch_size);

bool endswith(const std::string &str, const std::string &ending);

std::string getSuffix(const std::string& filePath);

std::string getFileName(const std::string& file_path, bool with_ext);

std::vector<std::string> splitString(const std::string &str, const std::string &delim);

void reIndexResults(std::vector<Result>& infer_results, std::map<int, std::string>& src_map, std::map<std::string, int>& tgt_map);

std::map<int, std::string> readFileToMap(const std::string &filePath);

std::string replaceSuffix(const std::string& path, const std::string& oldSuffix, const std::string& newSuffix);

std::string get_logfile_name(std::string& log_dir);

bool save_file(const std::string& file, const void* data, size_t length);

std::vector<unsigned char> load_file(const std::string& file);
