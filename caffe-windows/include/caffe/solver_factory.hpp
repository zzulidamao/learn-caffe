/**
 * @brief A solver factory that allows one to register solvers, similar to
 * layer factory. During runtime, registered solvers could be called by passing
 * a SolverParameter protobuffer to the CreateSolver function:
 *
 *     SolverRegistry<Dtype>::CreateSolver(param);
 *
 * There are two ways to register a solver. Assuming that we have a solver like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeSolver : public Solver<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Solver" at the end
 * ("MyAwesomeSolver" -> "MyAwesome").
 *
 * If the solver is going to be created simply by its constructor, in your C++
 * file, add the following line:
 *
 *    REGISTER_SOLVER_CLASS(MyAwesome);
 *
 * Or, if the solver is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Solver<Dtype*> GetMyAwesomeSolver(const SolverParameter& param) {
 *      // your implementation
 *    }
 *
 * then you can register the creator function instead, like
 *
 * REGISTER_SOLVER_CREATOR(MyAwesome, GetMyAwesomeSolver)
 *
 * Note that each solver type should only be registered once.
 */

#ifndef CAFFE_SOLVER_FACTORY_H_
#define CAFFE_SOLVER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Solver;

template <typename Dtype>
class SolverRegistry {
 public:
  typedef Solver<Dtype>* (*Creator)(const SolverParameter&);//Creator函数指针：指向返回值为Solver*,参数为SolverParameter的函数
  typedef std::map<string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry();

  // Adds a creator.
  static void AddCreator(const string& type, Creator creator);

  // Get a solver using a SolverParameter.
  static Solver<Dtype>* CreateSolver(const SolverParameter& param);

  static vector<string> SolverTypeList();

 private:
  // Solver registry should never be instantiated - everything is done with its
  // static variables.
  SolverRegistry();  // {}类的构造函数是私有的，所以不能创建类的对象，所有的成员函数都是static的属于类，通过类名直接调用

  static string SolverTypeListString();
};

template <typename Dtype>
class SolverRegisterer {
 public:
  SolverRegisterer(const string& type,
                   Solver<Dtype>* (*creator)(const SolverParameter&));
};
//调用SolverRegisterer的构造函数定义两个不同类型的静态成员变量
#define REGISTER_SOLVER_CREATOR(type, creator)                                 \
  static SolverRegisterer<float> g_creator_f_##type(#type, creator<float>);    \
  static SolverRegisterer<double> g_creator_d_##type(#type, creator<double>)   \
//多行宏定义
//定义一个函数为Creator_SGDSolver
//调用SGDSolver构造函数，并且返回对应的指针
#define REGISTER_SOLVER_CLASS(type)                                            \
  template <typename Dtype>                                                    \
  Solver<Dtype>* Creator_##type##Solver(                                       \
      const SolverParameter& param)                                            \
  {                                                                            \
    return new type##Solver<Dtype>(param);                                     \
  }                                                                            \
  REGISTER_SOLVER_CREATOR(type, Creator_##type##Solver)

}  // namespace caffe

#endif  // CAFFE_SOLVER_FACTORY_H_
