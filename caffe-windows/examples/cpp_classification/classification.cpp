#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp> //CMAKE 中定义编译宏,使用opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
//typedef 定义类型
typedef std::pair<string, float> Prediction;

class Classifier {
 public://公开的属性：类内类外都可以访问
  Classifier(const string& model_file,//网络结构文件
             const string& trained_file,//训练文件
             const string& mean_file,//均值文件
             const string& label_file);//标签文件

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);//分类函数：返回Prediction数组

 private://类内可以访问，类外不能访问，protected：类内可以访问，类外不能访问，但是子类可以访问，专门为子类服务
  void SetMean(const string& mean_file);//设置均值

  std::vector<float> Predict(const cv::Mat& img);//预测函数

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);//转换输入层？

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);//前处理函数

 private://类内不能访问，类外不能访问
  shared_ptr<Net<float> > net_;//智能指针指向Net网络结构，是float类型的
  cv::Size input_geometry_;//输入的尺寸
  int num_channels_;//通道数
  cv::Mat mean_;//均值图像
  std::vector<string> labels_;//标签
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));//智能指针reset：1.释放net_之前指向的地址	2.把net_从新指向新的地址
  net_->CopyTrainedLayersFrom(trained_file);//使用预训练网络模型初始化网络
  //num_inputs()输入的数量
  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";//判断网络是否只有一个输入，防止使用训练的网络结构文件进行预测
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];//输入层
  num_channels_ = input_layer->channels();//输入层通道数
  CHECK(num_channels_ == 3 || num_channels_ == 1)//单通道或者3通道
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());//输入层的尺寸

  /* Load the binaryproto mean file. */
  SetMean(mean_file);//设置均值

  /* Load labels. */
  std::ifstream labels(label_file.c_str());//加载标签文件
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));//读取标签文件中的每一行

  Blob<float>* output_layer = net_->output_blobs()[0];//输出层
  CHECK_EQ(labels_.size(), output_layer->channels())//输出层的输出数量要与标签数量对应起来主要为了分类
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;//从大到小排列
}

/* Return the indices of the top N values of vector v. 返回数组中的前N大的索引*/
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;//返回索引
}

/* Return the top N predictions.返回前N个预测结果 */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);//前N大的索引数组
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));//标签，置信度
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();//均值数据：图像均值：全部训练数据集中的数据各个通道对应像素相加求平均值
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);//初始化均值图像
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();//连续内存，步长为h*w，下一个通道的均值
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);//融合

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);//求像素平均：每个通道所有像素求出一个平均值
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
	std::cout << net_->input_blobs().size() << std::endl;
  Blob<float>* input_layer = net_->input_blobs()[0];//输入层
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);//1*3*224*224
  /* Forward dimension change to all layers. 根据输入层的尺寸去调整各个中间层的尺寸*/
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);//把输入层数据变换为mat

  Preprocess(img, &input_channels);//预处理

  net_->Forward();//前向传播

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];//输出层
  const float* begin = output_layer->cpu_data();//输出层数据
  const float* end = begin + output_layer->channels();//输出层的输出个数
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];//输入层blob

  int width = input_layer->width();
  int height = input_layer->height();//尺寸
  float* input_data = input_layer->mutable_cpu_data();//输入数据
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);//初始化输入数据到图像中
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network.把输入图像转换到网络输入层需要的格式 */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)//如果图像为3通道，输入层为单通道
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);//转换为灰度图
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);//如果输入图像不满足网络输入层的尺寸resize操作
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)//转为float32
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;//减去均值
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);
  //判断转换是否成功
  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv) {
  /*if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg" << std::endl;
    return 1;
  }*/

  ::google::InitGoogleLogging(argv[0]);
  
	string model_file   = "I:\\86\\Caffe_Manual-master\\learn_caffe\\source\\cifar10_full_deploy.prototxt";
    string trained_file = "I:\\86\\Caffe_Manual-master\\learn_caffe\\source\\cifar10_full_iter_1000.caffemodel";
    string label_file   = "I:\\86\\Caffe_Manual-master\\learn_caffe\\source\\labels.txt";
    string file     = "I:\\86\\Caffe_Manual-master\\learn_caffe\\source\\cifar-10\\test\\0\\0_10.png";
    string mean_file    = "I:\\86\\Caffe_Manual-master\\learn_caffe\\source\\mean.binaryproto";
  
  /*string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file    = argv[3];
  string label_file   = argv[4];*/
  Classifier classifier(model_file, trained_file, mean_file, label_file);

  //string file = argv[5];

  std::cout << "---------- Prediction for "
            << file << " ----------" << std::endl;

  cv::Mat img = cv::imread(file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file;
  std::vector<Prediction> predictions = classifier.Classify(img);

  /* Print the top N predictions. */
  for (size_t i = 0; i < predictions.size(); ++i) {
    Prediction p = predictions[i];//固定输出的小数点位数
    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
              << p.first << "\"" << std::endl;
  }
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
