/*���ļ�ʹ��ѵ���õ�ģ�����������ݵõ��м����������ͼ�����浽����Ӳ��*/
/*
extract_features \   //��ִ�еĳ���
pretrained_net_param \ //Ԥѵ������.caffemodel
feature_extraction_proto_file \ //���������ļ�  .prototxt
extract_feature_blob_name1 [, name2, ...] \   //��Ҫ��ȡ��blob�����������������
save_feature_dataset_name1 [, name2, ...] \  //����������
num_mini_batches \  //��������ȡ������������Ŀ
db_type \ //��������ݵĸ�ʽ��lmdb����leveldb
CPU/GPU \ //ѡ��һ����ģʽ���У�CPUģʽ����GPUģʽ
device_id=0   //���ʹ��GPU����ѡ���豸���
*/
//�ο���https://blog.csdn.net/seu_nuaa_zc/article/details/80549540
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
int feature_extraction_pipeline(int argc, char** argv);//����ģ������

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {//����
  ::google::InitGoogleLogging(argv[0]);//��ʼ����־
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
  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {//�Ƚ�arg_pos�����Ƿ�ΪGPU
    LOG(ERROR)<< "Using GPU";
    int device_id = 0;
    if (argc > arg_pos + 1) {//ָ��GPU���
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
  std::string pretrained_binary_proto(argv[++arg_pos]);//��һ������Ϊģ�����ƣ�ģ�Ͳ���

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
  std::string feature_extraction_proto(argv[++arg_pos]);//�ڶ�������Ϊ����ṹ
  boost::shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(feature_extraction_proto, caffe::TEST));//��������Ϊtest
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);//ʹ��ѵ���õ�ģ�ͳ�ʼ������

  std::string extract_feature_blob_names(argv[++arg_pos]);//blob�����ƣ��ã�����
  std::vector<std::string> blob_names;
  boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));

  std::string save_feature_dataset_names(argv[++arg_pos]);//�����������ƣ��ã��Ÿ���
  std::vector<std::string> dataset_names;
  boost::split(dataset_names, save_feature_dataset_names,
               boost::is_any_of(","));
  CHECK_EQ(blob_names.size(), dataset_names.size()) <<
      " the number of blob names and dataset names must be equal";//blob������Ҫ�뱣�����ݿ������һ��
  size_t num_features = blob_names.size();

  for (size_t i = 0; i < num_features; i++) {//�ж�����ṹ���Ƿ��и�blob�����ƣ�
    CHECK(feature_extraction_net->has_blob(blob_names[i]))
        << "Unknown feature blob name " << blob_names[i]
        << " in the network " << feature_extraction_proto;
  }

  int num_mini_batches = atoi(argv[++arg_pos]);//batch size������

  std::vector<boost::shared_ptr<db::DB> > feature_dbs;
  std::vector<boost::shared_ptr<db::Transaction> > txns;
  const char* db_type = argv[++arg_pos];//���ݿ�����
  for (size_t i = 0; i < num_features; ++i) {
    LOG(INFO)<< "Opening dataset " << dataset_names[i];
    boost::shared_ptr<db::DB> db(db::GetDB(db_type));
    db->Open(dataset_names.at(i), db::NEW);
    feature_dbs.push_back(db);
    boost::shared_ptr<db::Transaction> txn(db->NewTransaction());
    txns.push_back(txn);
  }

  LOG(ERROR)<< "Extracting Features";
  //���ӣ��Ե�һ���������п��ӻ�����һ�������"conv1"��ά����Ϣ��96*3*11*11����96������ˣ�ÿ���������3ͨ���ģ�ÿ������˳ߴ�Ϊ11*11    
  //�ʸþ������96��ͼ��ÿ��ͼ��11*11����ͨ��BGRͼ��
  Datum datum;
  std::vector<int> image_indices(num_features, 0);
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {//���ѭ��batch size
    feature_extraction_net->Forward();//ǰ�򴫲����õ�ÿһ�������
    for (int i = 0; i < num_features; ++i) {//��Ҫ��ȡ����������
      const boost::shared_ptr<Blob<Dtype> > feature_blob =
        feature_extraction_net->blob_by_name(blob_names[i]);//ͨ�������ڴ���֮��������еõ�����blob
      int batch_size = feature_blob->num();//blob��bs?
      int dim_features = feature_blob->count() / batch_size;//ÿ��ͼ������ά��
      const Dtype* feature_blob_data;
      for (int n = 0; n < batch_size; ++n) {
        datum.set_height(feature_blob->height());
        datum.set_width(feature_blob->width());
        datum.set_channels(feature_blob->channels());
        datum.clear_data();
        datum.clear_float_data();
        feature_blob_data = feature_blob->cpu_data() +
            feature_blob->offset(n);//btch����Դ���е�λ��+���ڵĴ��λ��
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
