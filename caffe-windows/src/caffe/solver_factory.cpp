#include <string>
#include <vector>

#include "caffe/solver_factory.hpp"

namespace caffe {

template <typename Dtype>
typename SolverRegistry<Dtype>::CreatorRegistry&
SolverRegistry<Dtype>::Registry() {
  static CreatorRegistry* g_registry_ = new CreatorRegistry;//因为变量g_registry_是static的即使多次调用，也只会定义一次
  return *g_registry_;
}

template <typename Dtype>
void SolverRegistry<Dtype>::AddCreator(const string& type, Creator creator) {
  CreatorRegistry& registry = Registry();//多次调用Registry，也只会定义一个g_registry_，然后把新的函数指针添加到g_registry_中
  CHECK_EQ(registry.count(type), 0) << "Solver type " << type
                                    << " already registered.";
  registry[type] = creator;
}

// Get a solver using a SolverParameter.
template <typename Dtype>
Solver<Dtype>* SolverRegistry<Dtype>::CreateSolver(
    const SolverParameter& param) {
  const string& type = param.type();//根据param的type来生成对应的子类 SGD
  CreatorRegistry& registry = Registry();//CreatorRegistry对象map
  CHECK_EQ(registry.count(type), 1)//判断计数次数是否等于1
      << "Unknown solver type: " << type
      << " (known types: " << SolverTypeListString() << ")";
  return registry[type](param);//根据type返回注册过的slover* 并且(param)调用
}

template <typename Dtype>
vector<string> SolverRegistry<Dtype>::SolverTypeList() {
  CreatorRegistry& registry = Registry();
  vector<string> solver_types;
  for (typename CreatorRegistry::iterator iter = registry.begin();
       iter != registry.end(); ++iter) {
    solver_types.push_back(iter->first);
  }
  return solver_types;
}

// Solver registry should never be instantiated - everything is done with its
// static variables.
template <typename Dtype>
SolverRegistry<Dtype>::SolverRegistry() {}

template <typename Dtype>
string SolverRegistry<Dtype>::SolverTypeListString() {
  vector<string> solver_types = SolverTypeList();
  string solver_types_str;
  for (vector<string>::iterator iter = solver_types.begin();
       iter != solver_types.end(); ++iter) {
    if (iter != solver_types.begin()) {
      solver_types_str += ", ";
    }
    solver_types_str += *iter;
  }
  return solver_types_str;
}
//在构造函数中调用AddCreator，把创建的指针对象放到全局map中
template <typename Dtype>
SolverRegisterer<Dtype>::SolverRegisterer(
    const string& type, Solver<Dtype>* (*creator)(const SolverParameter&)) {
  SolverRegistry<Dtype>::AddCreator(type, creator);
}

INSTANTIATE_CLASS(SolverRegistry);
INSTANTIATE_CLASS(SolverRegisterer);

}  // namespace caffe
