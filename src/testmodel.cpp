#include <tensorflow/core/framework/tensor.h>

#include "MyModel.hpp"

int main(int argc, char** argv)
{
    std::string export_dir = "/mnt/model_info_test/savetf_loadtf/output";
    if(argc > 2){
        std::cout << "Invalid number of arguments\n";
        return -1;
    } else if (argc == 2)
    {
        export_dir = std::string(argv[1]);
    }

    MyModel model(export_dir);
    model.printAllSignatureInfo();

    std::vector<tensorflow::Tensor> outputs;
    model.predict(&outputs);
    model.train();
    model.predict(&outputs);
    std::cout << std::endl;
    return 0;
}
