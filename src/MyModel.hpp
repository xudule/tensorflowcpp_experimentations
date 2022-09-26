#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>
#include "tensorflow/cc/framework/ops.h"

using namespace tensorflow;
using namespace std;

class MyModel
{
public:
    MyModel();
    ~MyModel();

    void printAllSignatureInfo();
    bool predict(std::vector<tensorflow::Tensor>*);
    bool train();
private:
    SavedModelBundle bundle_;
    protobuf::Map<string, SignatureDef> sig_;
    Session* session_;
};