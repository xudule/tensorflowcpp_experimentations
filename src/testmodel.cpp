#include <tensorflow/core/framework/tensor.h>

#include "MyModel.hpp"

int main() {
    MyModel model;
    model.printAllSignatureInfo();

    std::vector<tensorflow::Tensor>* outputs;
    model.predict(outputs);
    model.train();
    model.predict(outputs);
    return 0;
}
