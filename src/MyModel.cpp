#include "MyModel.hpp"

//constexpr 
using namespace tensorflow;
using namespace std;

void printSeparated(string s){
    cout << string(40, '-') << s << string(40, '-') << "\n";
}

MyModel::MyModel()
{
    //TODO: make model path as a program parameter
    const std::string export_dir = "/mnt/model_info_test/savetf_loadtf/output";
    SessionOptions session_options;
    RunOptions run_options;
    Status s = LoadSavedModel(session_options, run_options, export_dir, {kSavedModelTagServe},
                &bundle_);
    std::cout << "Loading model status is: " << s << '\n';
    //TODO: check status.
    //TODO: maybe move session out of constructor
    sig_ = bundle_.GetSignatures();
    session_ = bundle_.GetSession();
}
MyModel::~MyModel()
{
    Status s = session_->Close();
    std::cout << "Session closed with status: " << s << endl;
}

//TODO: make input as a parameter
bool MyModel::predict(std::vector<tensorflow::Tensor>* outputs)
{
    //TODO add exception
    printSeparated("Prediction");
    SignatureDef serv_sig = sig_.at("serving_default");
    TensorInfo i_tensor = serv_sig.inputs().at("inputs");
    TensorInfo o_tensor = serv_sig.outputs().at("output_0");

    Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,1}));
    input_tensor.matrix<float>()(0) = 10.0f;
    std::vector<std::pair<std::string, Tensor> > input = {{i_tensor.name(), input_tensor}};

    Status s = session_->Run(input, {o_tensor.name()}, {}, outputs);
    if (!s.ok())
    {
	    cout << "session NOK\n";
	    cout << s << endl;
        return false;
    }
    else
    {
	    cout << "session OK " << s << std::endl;
        auto output_tensor = outputs->at(0).flat<float>();
	    cout << "output is " << output_tensor << endl;
        return true;
    }
}

bool MyModel::train()
{
    printSeparated("Training");
    SignatureDef train_sig = sig_.at("my_train");
    TensorInfo o_tensor = train_sig.outputs().at("output_0");
    std::vector<tensorflow::Tensor> my_outputs;
    Status s = session_->Run({}, {o_tensor.name()}, {}, &my_outputs);
    if (!s.ok())
    {
	    cout << "session NOK\n";
	    cout << s << endl;
        return false;
    }
    else
    {
	    cout << "session OK " << s << std::endl;
        return true;
    }
}

void MyModel::printAllSignatureInfo()
{
    printSeparated("All signatures of model");
    for(auto it=sig_.begin(); it != sig_.end(); it++)
    {
    	cout << "signature: " << it->first << endl;
        google::protobuf::Map<std::string, TensorInfo> inputs = it->second.inputs();
        for (auto it2=inputs.begin(); it2!=inputs.end(); it2++)
        {
            cout << "input name: " << it2->first << endl;
            cout << "tensorinfo name "<<it2->second.name() << endl;
        }
        google::protobuf::Map<std::string, TensorInfo> outputs = it->second.outputs();
        for (auto it2=outputs.begin(); it2!=outputs.end(); it2++)
        {
            cout << "output name: " << it2->first << endl;
            cout << "tensorinfo name "<<it2->second.name() << endl;
        }
        cout << '\n';
    }
}
