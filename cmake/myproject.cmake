file(GLOB ${PROJECT_NAME}_EXE_SRC
    src/*.cpp
)

include_directories(
/tensorflow_src/bazel-bin/tensorflow/include/
/tensorflow_src/bazel-bin/tensorflow/include/src/
)

find_library(TF_LIB tensorflow_cc HINTS /tensorflow_src/bazel-bin/tensorflow/)
find_library(TF_FW_LIB tensorflow_framework HINTS /tensorflow_src/bazel-bin/tensorflow/)


add_executable(testmodel ${${PROJECT_NAME}_EXE_SRC})
target_link_libraries(testmodel ${TF_LIB} ${TF_FW_LIB})
