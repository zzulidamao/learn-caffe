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
//��̬��Ա������ڱ���׶η���ռ䣬�洢��ȫ��������̬��Ա�����������������ҽ���һ�Σ���������ж����ʼ������̬��Ա����������ĳ�����󣬲�ռ�ö�����ڴ�ռ䣬����ĳ���࣬���ж�����
//��̬��Ա����û��thisָ�룬�����޷����ʷǾ�̬��Ա�������ڶ���û�б�����ǰ�Ϳ���ͨ�������ƽ��е��á���Ҫ���ڹ���̬��Ա����
namespace caffe {
//����Layer��ģ��
template <typename Dtype>
class Layer;
//��ע����ģ��
template <typename Dtype>
class LayerRegistry {
 public://���庯��ָ�룺��������shared_ptr<Layer<Dtype>>������������const LayerParameter��-- shared_ptr<<>>(*)(),����ָ�����Creator==func
  typedef shared_ptr<Layer<Dtype> > (*Creator)(const LayerParameter&);
  typedef std::map<string, Creator> CreatorRegistry;//map:string,����ָ��

  static CreatorRegistry& Registry();//����һ������ֵΪmap�ľ�̬����

  // Adds a creator.
  static void AddCreator(const string& type, Creator creator);//��̬����ֻ�ܷ��ʾ�̬��Ա����

  // Get a layer using a LayerParameter.
  static shared_ptr<Layer<Dtype> > CreateLayer(const LayerParameter& param);

  static vector<string> LayerTypeList();
  //˽�г�Ա������ֻ�������ڽ��з��ʣ����ⲻ�ܷ��ʡ�����Ҳ���ܷ���
 private:
  // Layer registry should never be instantiated - everything is done with its
  // static variables.
  LayerRegistry();//���캯��

  static string LayerTypeListString();
};

template <typename Dtype>
class LayerRegisterer {
 public:
  LayerRegisterer(const string& type,
                  shared_ptr<Layer<Dtype> > (*creator)(const LayerParameter&));//���캯������AddCreator
};
//ע������float/double��̬��Ա����--���ù��캯��
#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \
//���к궨�壬��ͬ�Ĳ㲻ͬ����
#define REGISTER_LAYER_CLASS(type)                                             \
  template <typename Dtype>                                                    \
  shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param) \
  {                                                                            \
    return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));           \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_H_
