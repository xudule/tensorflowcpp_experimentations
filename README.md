# tensorflowcpp_experimentations
This repository represents some experimentation with tensorflow library.

# Requirements:
Compile tensorflow c++ library: Currently I use tensorflow dev container to generate the tensorflow c++ library. It's generated under /tensorflow_src/bazel-bin/tensorflow/. Headers are under: /tensorflow_src/bazel-bin/tensorflow/include.

If you put your library and include files in different folder, you need to change the cmake/myproject.cmake

# Steps to run the program:
- Generate model with model/linear_model.py
- compile testmodel binary