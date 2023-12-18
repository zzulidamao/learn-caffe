/*该文件使用训练好的模型与输入数据得到中间推理的特征图并保存到本地硬盘*/
/*
extract_features \   //可执行的程序
pretrained_net_param \ //预训练网络.caffemodel
feature_extraction_proto_file \ //网络描述文件  .prototxt
extract_feature_blob_name1 [, name2, ...] \   //需要提取的blob名（层输入输出名）
save_feature_dataset_name1 [, name2, ...] \  //保存特征名
num_mini_batches \  //做特征提取的数据批量数目
db_type \ //输入的数据的格式，lmdb或者leveldb
CPU/GPU \ //选择一何种模式运行，CPU模式或者GPU模式
device_id=0   //如果使用GPU，则选择设备编号
*/
//参考：https://blog.csdn.net/seu_nuaa_zc/article/details/80549540
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;
namespace db = caffe::db;

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);//函数模板声明

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {//定义
  ::google::InitGoogleLogging(argv[0]);//初始化日志
  const int num_required_args = 7;
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "This program takes in a trained network and an input data layer, and then"
    " extract features of the input data produced by the net.\n"
    "Usage: extract_features  pretrained_net_param"
    "  feature_extraction_proto_file  extract_feature_blob_name1[,name2,...]"
    "  save_feature_dataset_name1[,name2,...]  num_mini_batches  db_type"
    "  [CPU/GPU] [DEVICE_ID=0]\n"
    "Note: you can extract multiple features in one pass by specifying"
    " multiple feature blob names and dataset names separated by ','."
    " The names cannot contain white space characters and the number of blobs"
    " and datasets must be equal.";
    return 1;
  }
  int arg_pos = num_required_args;

  arg_pos = num_required_args;
  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {//比较arg_pos参数是否为GPU
    LOG(ERROR)<< "Using GPU";
    int device_id = 0;
    if (argc > arg_pos + 1) {//指定GPU编号
      device_id = atoi(argv[arg_pos + 1]);
      CHECK_GE(device_id, 0);
    }
    LOG(ERROR) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  arg_pos = 0;  // the name of the executable
  std::string pretrained_binary_proto(argv[++arg_pos]);//第一个参数为模型名称，模型参数

  // Expected prototxt contains at least one data layer such as
  //  the layer data_layer_name and one feature blob such as the
  //  fc7 top blob to extract features.
  /*
   layers {
     name: "data_layer_name"
     type: DATA
     data_param {
       source: "/path/to/your/images/to/extract/feature/images_leveldb"
       mean_file: "/path/to/your/image_mean.binaryproto"
       batch_size: 128
       crop_size: 227
       mirror: false
     }
     top: "data_blob_name"
     top: "label_blob_name"
   }
   layers {
     name: "drop7"
     type: DROPOUT
     dropout_param {
       dropout_ratio: 0.5
     }
     bottom: "fc7"
     top: "fc7"
   }
   */
  std::string feature_extraction_proto(argv[++arg_pos]);//第二个参数为网络结构
  boost::shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(feature_extraction_proto, caffe::TEST));//定义网络为test
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);//使用训练好的模型初始化网络

  std::string extract_feature_blob_names(argv[++arg_pos]);//blob的名称：用，隔开
  std::vector<std::string> blob_names;
  boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));

  std::string save_feature_dataset_names(argv[++arg_pos]);//保存特征名称，用，号隔开
  std::vector<std::string> dataset_names;
  boost::split(dataset_names, save_feature_dataset_names,
               boost::is_any_of(","));
  CHECK_EQ(blob_names.size(), dataset_names.size()) <<
      " the number of blob names and dataset names must be equal";//blob的数量要与保存数据库的数量一致
  size_t num_features = blob_names.size();

  for (size_t i = 0; i < num_features; i++) {//判断网络结构中是否有该blob（名称）
    CHECK(feature_extraction_net->has_blob(blob_names[i]))
        << "Unknown feature blob name " << blob_names[i]
        << " in the network " << feature_extraction_proto;
  }

  int num_mini_batches = atoi(argv[++arg_pos]);//batch size的数量

  std::vector<boost::shared_ptr<db::DB> > feature_dbs;
  std::vector<boost::shared_ptr<db::Transaction> > txns;
  const char* db_type = argv[++arg_pos];//数据库类型
  for (size_t i = 0; i < num_features; ++i) {
    LOG(INFO)<< "Opening dataset " << dataset_names[i];
    boost::shared_ptr<db::DB> db(db::GetDB(db_type));
    db->Open(dataset_names.at(i), db::NEW);
    feature_dbs.push_back(db);
    boost::shared_ptr<db::Transaction> txn(db->NewTransaction());
    txns.push_back(txn);
  }

  LOG(ERROR)<< "Extracting Features";
  //例子：对第一个卷积层进行可视化，第一个卷积层"conv1"的维度信息是96*3*11*11，即96个卷积核，每个卷积核是3通道的，每个卷积核尺寸为11*11    
  //故该卷积层有96个图，每个图是11*11的三通道BGR图像
  Datum datum;
  std::vector<int> image_indices(num_features, 0);
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {//外层循环batch size
    feature_extraction_net->Forward();//前向传播：得到每一层的特征
    for (int i = 0; i < num_features; ++i) {//需要提取的特征数量
      const boost::shared_ptr<Blob<Dtype> > feature_blob =
        feature_extraction_net->blob_by_name(blob_names[i]);//通过名称在传播之后的网络中得到特征blob
      int batch_size = feature_blob->num();//blob的bs?
      int dim_features = feature_blob->count() / batch_size;//每个图的特征维度
      const Dtype* feature_blob_data;
      for (int n = 0; n < batch_size; ++n) {
        datum.set_height(feature_blob->height());
        datum.set_width(feature_blob->width());
        datum.set_channels(feature_blob->channels());
        datum.clear_data();
        datum.clear_float_data();
        feature_blob_data = feature_blob->cpu_data() +
            feature_blob->offset(n);//btch中相对大格中的位置+所在的大格位置
        for (int d = 0; d < dim_features; ++d) {
          datum.add_float_data(feature_blob_data[d]);
        }
        string key_str = caffe::format_int(image_indices[i], 10);

        string out;
        CHECK(datum.SerializeToString(&out));
        txns.at(i)->Put(key_str, out);
        ++image_indices[i];
        if (image_indices[i] % 1000 == 0) {
          txns.at(i)->Commit();
          txns.at(i).reset(feature_dbs.at(i)->NewTransaction());
          LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
              " query images for feature blob " << blob_names[i];
        }
      }  // for (int n = 0; n < batch_size; ++n)
    }  // for (int i = 0; i < num_features; ++i)
  }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
  // write the last batch
  for (int i = 0; i < num_features; ++i) {
    if (image_indices[i] % 1000 != 0) {
      txns.at(i)->Commit();
    }
    LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
        " query images for feature blob " << blob_names[i];
    feature_dbs.at(i)->Close();
  }

  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}
