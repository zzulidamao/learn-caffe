#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"
//使用命名空间caffe下的属性
using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;
//DEFINE_xxx()就是添加命令行的optional argument（--可选参数）
//而FLAGS_xxx可以从对应的命令行参数取出参数，解析命令行参数。相当于全局变量
//DEFINE_xxx(变量名称，默认值，帮助说明)
DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(phase, "",
    "Optional; network phase (TRAIN or TEST). Only used for 'time'.");
DEFINE_int32(level, 0,
    "Optional; network level.");
DEFINE_string(stage, "",
    "Optional; network stages (not to be confused with phase), "
    "separated by ','.");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
DEFINE_string(sigint_effect, "stop",
             "Optional; action to take when a SIGINT signal is received: "
              "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
             "Optional; action to take when a SIGHUP signal is received: "
             "snapshot, stop or none.");

// A simple registry for caffe commands.
//这里使用typedef 定义了一个函数指针：指向一个返回值为int，参数为空的函数；指针变量的名称为：BrewFunction,即BrewFunction === int (*)()
//函数名带括号的就是函数指针
typedef int (*BrewFunction)();
//这里使用typedef 定义了一个全局map
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

//多行宏定义每行结束用 "\" 
//宏定义中的“##”是分隔链接方式(用传入进来的函数名替换##func)：当func == train ==>class __Registerer_train 
//然后定义实现无参构造函数__Registerer_train()
//然后__Registerer_train g_registerer_train定义成员变量：
//调用无参构造函数，给g_brew_map["train"] = &train赋值;取函数地址
//宏·的优点1.类型无关的，性能优于函数调用，复用性强
//宏的缺点：在编译阶段进行展开替换，不方便调试，没有类型安全检查，运算符优先级问题
#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
	LOG(INFO) << "#func:" << #func; \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}
//map count(key) 如果map中存在key则返回1,否则返回0
//根据key从g_brew_map返回函数指针：train test
static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;//遍历输出中map中已经注册的函数指针
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// Parse GPU ids or use all available devices
//static 静态函数：在函数的返回值之前加上static关键字，这个函数只能在声明它的源文件（caffe.cpp）中使用，不能在其他的源文件中被使用，因此可以在其他的源文件中声明同名的函数不会引起冲突
//非静态函数，其实声明的时候，默认是extern就是可以被其他源文件使用，只是有很多时候书写省略了extern
static void get_gpus(vector<int>* gpus) {
  if (FLAGS_gpu == "all") {//FLAGS_gpu 通过DEFINE_string定义的命令行赋值的变量
    int count = 0;
#ifndef CPU_ONLY//不用cpu
    CUDA_CHECK(cudaGetDeviceCount(&count));//通过引用的方式获取可用cuda数量
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);//将可用gpu编号存到形参数组中
    }
  } else if (FLAGS_gpu.size()) {//如果FLAGS_gpu命令行参数不为空 0,1,2
    vector<string> strings;
    boost::split(strings, FLAGS_gpu, boost::is_any_of(","));//使用逗号对gpu编号字符串进行分割
    for (int i = 0; i < strings.size(); ++i) {
      gpus->push_back(boost::lexical_cast<int>(strings[i]));//boost::lexical_cast标准类型转换函数，推荐使用
    }
  } else {
    CHECK_EQ(gpus->size(), 0);//判断实参是否为空
  }
}

// Parse phase from flags
//解析是Train or Test
caffe::Phase get_phase_from_flags(caffe::Phase default_value) {
  if (FLAGS_phase == "")
    return default_value;
  if (FLAGS_phase == "TRAIN")
    return caffe::TRAIN;
  if (FLAGS_phase == "TEST")
    return caffe::TEST;
  LOG(FATAL) << "phase must be \"TRAIN\" or \"TEST\"";
  return caffe::TRAIN;  // Avoid warning
}

// Parse stages from flags
//分解FLAGS_stage
vector<string> get_stages_from_flags() {
  vector<string> stages;
  boost::split(stages, FLAGS_stage, boost::is_any_of(","));
  return stages;
}
//可以自己写一个函数，通过RegisterBrewFunction(action);进行注册，就像下面的device_query，等等一样
// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
  vector<int> gpus;
  get_gpus(&gpus);
  for (int i = 0; i < gpus.size(); ++i) {
    caffe::Caffe::SetDevice(gpus[i]);
    caffe::Caffe::DeviceQuery();
  }
  return 0;
}
RegisterBrewFunction(device_query);//宏是在预处理（编译）阶段进行展开的，因此在main函数之前会在map中生成4个键值对

// Load the weights from the specified caffemodel(s) into the train and test nets.
// caffe::Solver类要重点分析
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );//分解多个模型名称
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);//通过模型名称初始化网络层
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);//初始化测试网络的各个层
    }
  }
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
// 枚举作为函数返回值类型，只能返回枚举中定义的枚举常量
// 停止训练，或者使用之前的snapshot接着训练
caffe::SolverAction::Enum GetRequestedAction(
    const std::string& flag_value) {
  if (flag_value == "stop") {
    return caffe::SolverAction::STOP;
  }
  if (flag_value == "snapshot") {
    return caffe::SolverAction::SNAPSHOT;
  }
  if (flag_value == "none") {
    return caffe::SolverAction::NONE;
  }
  LOG(FATAL) << "Invalid signal effect \""<< flag_value << "\" was specified";
  return caffe::SolverAction::NONE;
}

// Train / Finetune a model.
int train() {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";//检测命令行参数是否大于0，即不为空
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())//检测snapshot与weight只用给一个就可以了，不用都给
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";
  vector<string> stages = get_stages_from_flags();

  caffe::SolverParameter solver_param;//通过proto定义的类，caffe命名空间，caffe.proto
  caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);//读取slover

  solver_param.mutable_train_state()->set_level(FLAGS_level);//设置网络状态：train
  for (int i = 0; i < stages.size(); i++) {
    solver_param.mutable_train_state()->add_stage(stages[i]);
  }

  // If the gpus flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  //如果命令行参数FLAGS_gpu为空，并且solver中设置使用gpu。如果solver中有gpu编号，则把gpu编号赋值给FLAGS_gpu，否则默认使用0
  if (FLAGS_gpu.size() == 0
      && solver_param.has_solver_mode()
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
      if (solver_param.has_device_id()) {
          FLAGS_gpu = "" +
              boost::lexical_cast<string>(solver_param.device_id());
      } else {  // Set default GPU if unspecified
          FLAGS_gpu = "" + boost::lexical_cast<string>(0);
      }
  }

  vector<int> gpus;
  get_gpus(&gpus);//读取本地可用gpu编号
  if (gpus.size() == 0) {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  } else {
    ostringstream s;
    for (int i = 0; i < gpus.size(); ++i) {
      s << (i ? ", " : "") << gpus[i];//把gpu编号拼接0，1，2
    }
    LOG(INFO) << "Using GPUs " << s.str();
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    for (int i = 0; i < gpus.size(); ++i) {
      cudaGetDeviceProperties(&device_prop, gpus[i]);//打印gpu的设备信息
      LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
    }
#endif
    solver_param.set_device_id(gpus[0]);
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_count(gpus.size());
  }
  //信号
  caffe::SignalHandler signal_handler(
        GetRequestedAction(FLAGS_sigint_effect),//停止的信号
        GetRequestedAction(FLAGS_sighup_effect));//快照的函数
  //定义一个指向Solver<float>的共享指针，通过调用SolverRegistry的静态成员函数CreateSolver，返回Solver基类的指针，用于初始化智能指针solver
  //共享指针与泛型模板编程要加强学习
  shared_ptr<caffe::Solver<float> >
      solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
  //通过信号函数来处理ctrl+c与关闭ternamial的操作，回调函数
  solver->SetActionFunction(signal_handler.GetActionFunction());
  //如果使用snapshot与weight则通过Restore、CopyLayers读取已经训练好的网络参数
  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Restore(FLAGS_snapshot.c_str());
  } else if (FLAGS_weights.size()) {
    CopyLayers(solver.get(), FLAGS_weights);
  }

  LOG(INFO) << "Starting Optimization";
  if (gpus.size() > 1) {//多卡训练
#ifdef USE_NCCL
    caffe::NCCL<float> nccl(solver);
    nccl.Run(gpus, FLAGS_snapshot.size() > 0 ? FLAGS_snapshot.c_str() : NULL);
#else
    LOG(FATAL) << "Multi-GPU execution not available - rebuild with USE_NCCL";
#endif
  } else {//开始优化训练
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";
  return 0;
}
RegisterBrewFunction(train);


// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
  vector<string> stages = get_stages_from_flags();

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpus[0]);
    LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST, FLAGS_level, &stages);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(&iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";
  caffe::Phase phase = get_phase_from_flags(caffe::TRAIN);
  vector<string> stages = get_stages_from_flags();

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, phase, FLAGS_level, &stages);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(&initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
      << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;//日志记录到文件的同时输出到strerr
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));//版本信息，cmake定义的宏CAFFE_VERSION
  // Usage message.命令行帮助信息
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time");
  
  // Run tool or show usage.
  // 全局初始化:解析命令行输入参数，输入 argc==3，输出argc ==2 是
  // 因为gflags在初始化完 "-solver" "-snapshot" "-weights"之后会把他们从argc,argv中删除,所以只剩下两个参数。 
  // "-solver" "-snapshot" "-weights"它们被gflags初始化为标志FLAGS_solver FLAGS_snapshot FLAGS_weights等。
  caffe::GlobalInit(&argc, &argv);
  
  if (argc == 2) {
#ifdef WITH_PYTHON_LAYER
    try {
#endif
      return GetBrewFunction(caffe::string(argv[1]))();//返回函数指针，然后()调用函数
#ifdef WITH_PYTHON_LAYER
    } catch (bp::error_already_set) {
      PyErr_Print();
      return 1;
    }
#endif
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}
