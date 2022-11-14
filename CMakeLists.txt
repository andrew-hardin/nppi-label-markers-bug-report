project(label_bug LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)

add_executable(demo main.cu)
target_link_libraries(demo PUBLIC
  nppc
  nppif
  nppig)