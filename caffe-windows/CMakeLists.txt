cmake_minimum_required(VERSION 2.8.7)
if(MSVC)
  # CMake 3.4 introduced a WINDOWS_EXPORT_ALL_SYMBOLS target property that makes it possible to
  # build shared libraries without using the usual declspec() decoration.
  # See: https://blog.kitware.com/create-dlls-on-windows-without-declspec-using-new-cmake-export-all-feature/
  # and https://cmake.org/cmake/help/v3.5/prop_tgt/WINDOWS_EXPORT_ALL_SYMBOLS.html
  # for details.
  cmake_minimum_required(VERSION 3.4)
endif()
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

# ---[ Caffe project 使用C或者c++语言
project(Caffe C CXX)

# ---[ Caffe version
set(CAFFE_TARGET_VERSION "1.0.0" CACHE STRING "Caffe logical version")
set(CAFFE_TARGET_SOVERSION "1.0.0" CACHE STRING "Caffe soname version")
# 添加宏定义:版本管理，在代码中进行使用
add_definitions(-DCAFFE_VERSION=${CAFFE_TARGET_VERSION})

# ---[ Using cmake scripts and modules
# 把cmake脚本路径追加到CMAKE_MODULE_PATH中，PROJECT_SOURCE_DIR：cmakelists所在目录
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

include(ExternalProject)
include(GNUInstallDirs)
#把编译需要用到module.cmake包含进来
include(cmake/Utils.cmake)
include(cmake/Targets.cmake)
include(cmake/Misc.cmake)
include(cmake/Summary.cmake)
include(cmake/ConfigGen.cmake)
include(cmake/WindowsCreateLinkHeader.cmake)
include(cmake/TargetResolvePrerequesites.cmake)

# ---[ Options,caffe_option在cmake/Utils.cmake中定义的函数
# 构建命令可选参数，成为代码中的宏定义，在代码中使用
caffe_option(CPU_ONLY  "Build Caffe without CUDA support" OFF) # TODO: rename to USE_CUDA
caffe_option(USE_CUDNN "Build Caffe with cuDNN library support" ON IF NOT CPU_ONLY)
caffe_option(USE_NCCL "Build Caffe with NCCL library support" OFF)
if(MSVC)
  # default to static libs 不能使用vs编译动态库，编译命令行参数，使用cnake -D进行设置
  caffe_option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
else()
  caffe_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
endif()
caffe_option(BUILD_python "Build Python wrapper" ON)
set(python_version "2" CACHE STRING "Specify which Python version to use")
caffe_option(BUILD_matlab "Build Matlab wrapper" OFF)
caffe_option(BUILD_docs   "Build documentation" ON IF UNIX OR APPLE)
caffe_option(BUILD_python_layer "Build the Caffe Python layer" ON)
caffe_option(USE_OPENCV "Build with OpenCV support" ON) # 在编译过程中引入opencv,在代码中使用该宏
caffe_option(USE_LEVELDB "Build with levelDB" ON)
caffe_option(USE_LMDB "Build with lmdb" ON)
caffe_option(ALLOW_LMDB_NOLOCK "Allow MDB_NOLOCK when reading LMDB files (only if necessary)" OFF)
caffe_option(USE_OPENMP "Link with OpenMP (when your BLAS wants OpenMP and you get linker errors)" OFF)
caffe_option(protobuf_MODULE_COMPATIBLE "Make the protobuf-config.cmake compatible with the module mode" ON IF MSVC)
caffe_option(COPY_PREREQUISITES "Copy the prerequisites next to each executable or shared library directory" ON IF MSVC)
caffe_option(INSTALL_PREREQUISITES "Install the prerequisites next to each executable or shared library directory" ON IF MSVC)

if(MSVC AND BUILD_SHARED_LIBS)
  if(CMAKE_GENERATOR MATCHES "Visual Studio")
    # see issue https://gitlab.kitware.com/cmake/cmake/issues/16552#note_215236
    message(FATAL_ERROR "The Visual Studio generator cannot build a shared library. Use the Ninja generator instead.")
  endif()
  # Some tests (solver tests) fail when caffe is built as a shared library. The problem comes
  # from protobuf that has a global static empty_string_ variable. Since caffe and test.testbin
  # link to a static protobuf library both end up with their own instance of the empty_string_
  # variable. This causes some SEH exception to occur. In practice if the caffe executable does not link
  # to protobuf this problem should not happen. Use at your own risk.
  message(WARNING "Some tests (solvers) will fail when building as a shared library with MSVC")
endif()

# ---[ Prebuild dependencies on windows预先编译依赖库如glog等
include(cmake/WindowsDownloadPrebuiltDependencies.cmake)

# ---[ Dependencies编译第三方库
include(cmake/Dependencies.cmake)

# ---[ Flags 如果是unix或者苹果
if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()

caffe_set_caffe_link()
#标准c++版本
if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()

# ---[ Warnings 禁用编译警告：有符号数与无符号数的比较，未初始化
caffe_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)

# ---[ Config generation 文件配置函数：在下面定义，需要两个参数
configure_file(cmake/Templates/caffe_config.h.in "${PROJECT_BINARY_DIR}/caffe_config.h")

# ---[ Includes 设置caffe的头文件与源文件路径
set(Caffe_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(Caffe_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR})#添加包含路径

# ---[ Includes & defines for CUDA

# cuda_compile() does not have per-call dependencies or include pathes
# (cuda_compile() has per-call flags, but we set them here too for clarity)
#
# list(REMOVE_ITEM ...) invocations remove PRIVATE and PUBLIC keywords from collected definitions and include pathes
if(HAVE_CUDA) # HAVE_CUDA哪里来？
  # pass include pathes to cuda_include_directories()
  set(Caffe_ALL_INCLUDE_DIRS ${Caffe_INCLUDE_DIRS})
  list(REMOVE_ITEM Caffe_ALL_INCLUDE_DIRS PRIVATE PUBLIC)
  cuda_include_directories(${Caffe_INCLUDE_DIR} ${Caffe_SRC_DIR} ${Caffe_ALL_INCLUDE_DIRS})

  # add definitions to nvcc flags directly
  set(Caffe_ALL_DEFINITIONS ${Caffe_DEFINITIONS})
  list(REMOVE_ITEM Caffe_ALL_DEFINITIONS PRIVATE PUBLIC)
  list(APPEND CUDA_NVCC_FLAGS ${Caffe_ALL_DEFINITIONS})
endif()

# ---[ Subdirectories添加需要编译的子项目（子目录）
add_subdirectory(src/gtest)
add_subdirectory(src/caffe)
add_subdirectory(tools)
add_subdirectory(examples)
add_subdirectory(python)
add_subdirectory(matlab)
add_subdirectory(docs)

# ---[ Linter target
add_custom_target(lint COMMAND ${CMAKE_COMMAND} -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)

# ---[ pytest target
if(BUILD_python)# cmake -D设置
  if(UNIX)
    set(python_executable python${python_version})
  else()
    set(python_executable ${PYTHON_EXECUTABLE})
  endif()
  add_custom_target(pytest COMMAND ${python_executable} -m unittest discover -s caffe/test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python )
  add_dependencies(pytest pycaffe)
endif()

# ---[ uninstall target
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Uninstall.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/Uninstall.cmake
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/Uninstall.cmake)

# ---[ Configuration summary
caffe_print_configuration_summary()

# ---[ Export configs generation
caffe_generate_export_configs()

