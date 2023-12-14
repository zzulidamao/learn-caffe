/**
 * @brief A layer factory that allows one to register layers.
 * During runtime, registered layers can be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:
 *
 *     LayerRegistry<Dtype>::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like
 *
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.
 */

#ifndef CAFFE_LAYER_FACTORY_H_
#define CAFFE_LAYER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
//静态成员表变量在编译阶段分配空间，存储于全局区。静态成员变量类内声明（有且仅有一次），类外进行定义初始化。静态成员变量不属于某个对象，不占用对象的内存空间，属于某个类，所有对象共享。
//静态成员函数没有this指针，所以无法访问非静态成员变量。在对象没有被创建前就可以通过类名称进行调用。主要用于管理静态成员变量
namespace caffe {
//声明Layer类模板
template <typename Dtype>
class Layer;
//层注册类模板
template <typename Dtype>
class LayerRegistry {
 public://定义函数指针：返回类型shared_ptr<Layer<Dtype>>，函数参数（const LayerParameter）-- shared_ptr<<>>(*)(),函数指针变量Creator==func
  typedef shared_ptr<Layer<Dtype> > (*Creator)(const LayerParameter&);
  typedef std::map<string, Creator> CreatorRegistry;//map:string,函数指针

  static CreatorRegistry& Registry();//声明一个返回值为map的静态函数

  // Adds a creator.
  static void AddCreator(const string& type, Creator creator);//静态函数只能访问静态成员变量

  // Get a layer using a LayerParameter.
  static shared_ptr<Layer<Dtype> > CreateLayer(const LayerParameter& param);

  static vector<string> LayerTypeList();
  //私有成员函数：只能在类内进行访问，类外不能访问。子类也不能访问
 private:
  // Layer registry should never be instantiated - everything is done with its
  // static variables.
  LayerRegistry();//构造函数

  static string LayerTypeListString();
};

template <typename Dtype>
class LayerRegisterer {
 public:
  LayerRegisterer(const string& type,
                  shared_ptr<Layer<Dtype> > (*creator)(const LayerParameter&));//构造函数调用AddCreator
};
//注册两个float/double静态成员变量--调用构造函数
#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \
//多行宏定义，不同的层不同的类
#define REGISTER_LAYER_CLASS(type)                                             \
  template <typename Dtype>                                                    \
  shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param) \
  {                                                                            \
    return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));           \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_H_
