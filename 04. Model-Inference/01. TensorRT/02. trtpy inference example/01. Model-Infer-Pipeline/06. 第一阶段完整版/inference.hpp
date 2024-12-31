#ifndef INFERENCE_HPP
#define INFERENCE_HPP
#include <memory>
#include <vector>
#include <string>


class InferInterface
{
public:
    virtual std::vector<uint8_t> forward(std::vector<uint8_t>& input_image_bytes) = 0;
};
std::shared_ptr<InferInterface> create_infer(const std::string &file);

#endif //INFERENCE_HPP
