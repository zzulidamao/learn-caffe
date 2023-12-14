#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp> //CMAKE �ж�������,ʹ��opencv
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
//typedef ��������
typedef std::pair<string, float> Prediction;

class Classifier {
 public://���������ԣ��������ⶼ���Է���
  Classifier(const string& model_file,//����ṹ�ļ�
             const string& trained_file,//ѵ���ļ�
             const string& mean_file,//��ֵ�ļ�
             const string& label_file);//��ǩ�ļ�

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);//���ຯ��������Prediction����

 private://���ڿ��Է��ʣ����ⲻ�ܷ��ʣ�protected�����ڿ��Է��ʣ����ⲻ�ܷ��ʣ�����������Է��ʣ�ר��Ϊ�������
  void SetMean(const string& mean_file);//���þ�ֵ

  std::vector<float> Predict(const cv::Mat& img);//Ԥ�⺯��

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);//ת������㣿

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);//ǰ������

 private://���ڲ��ܷ��ʣ����ⲻ�ܷ���
  shared_ptr<Net<float> > net_;//����ָ��ָ��Net����ṹ����float���͵�
  cv::Size input_geometry_;//����ĳߴ�
  int num_channels_;//ͨ����
  cv::Mat mean_;//��ֵͼ��
  std::vector<string> labels_;//��ǩ
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
  net_.reset(new Net<float>(model_file, TEST));//����ָ��reset��1.�ͷ�net_֮ǰָ��ĵ�ַ	2.��net_����ָ���µĵ�ַ
  net_->CopyTrainedLayersFrom(trained_file);//ʹ��Ԥѵ������ģ�ͳ�ʼ������
  //num_inputs()���������
  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";//�ж������Ƿ�ֻ��һ�����룬��ֹʹ��ѵ��������ṹ�ļ�����Ԥ��
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];//�����
  num_channels_ = input_layer->channels();//�����ͨ����
  CHECK(num_channels_ == 3 || num_channels_ == 1)//��ͨ������3ͨ��
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());//�����ĳߴ�

  /* Load the binaryproto mean file. */
  SetMean(mean_file);//���þ�ֵ

  /* Load labels. */
  std::ifstream labels(label_file.c_str());//���ر�ǩ�ļ�
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));//��ȡ��ǩ�ļ��е�ÿһ��

  Blob<float>* output_layer = net_->output_blobs()[0];//�����
  CHECK_EQ(labels_.size(), output_layer->channels())//�������������Ҫ���ǩ������Ӧ������ҪΪ�˷���
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;//�Ӵ�С����
}

/* Return the indices of the top N values of vector v. ���������е�ǰN�������*/
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;//��������
}

/* Return the top N predictions.����ǰN��Ԥ���� */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);//ǰN�����������
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));//��ǩ�����Ŷ�
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
  float* data = mean_blob.mutable_cpu_data();//��ֵ���ݣ�ͼ���ֵ��ȫ��ѵ�����ݼ��е����ݸ���ͨ����Ӧ���������ƽ��ֵ
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);//��ʼ����ֵͼ��
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();//�����ڴ棬����Ϊh*w����һ��ͨ���ľ�ֵ
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);//�ں�

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);//������ƽ����ÿ��ͨ�������������һ��ƽ��ֵ
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
	std::cout << net_->input_blobs().size() << std::endl;
  Blob<float>* input_layer = net_->input_blobs()[0];//�����
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);//1*3*224*224
  /* Forward dimension change to all layers. ���������ĳߴ�ȥ���������м��ĳߴ�*/
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);//����������ݱ任Ϊmat

  Preprocess(img, &input_channels);//Ԥ����

  net_->Forward();//ǰ�򴫲�

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];//�����
  const float* begin = output_layer->cpu_data();//���������
  const float* end = begin + output_layer->channels();//�������������
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];//�����blob

  int width = input_layer->width();
  int height = input_layer->height();//�ߴ�
  float* input_data = input_layer->mutable_cpu_data();//��������
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);//��ʼ���������ݵ�ͼ����
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network.������ͼ��ת���������������Ҫ�ĸ�ʽ */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)//���ͼ��Ϊ3ͨ���������Ϊ��ͨ��
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
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
    cv::resize(sample, sample_resized, input_geometry_);//�������ͼ���������������ĳߴ�resize����
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)//תΪfloat32
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;//��ȥ��ֵ
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);
  //�ж�ת���Ƿ�ɹ�
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
    Prediction p = predictions[i];//�̶������С����λ��
    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
              << p.first << "\"" << std::endl;
  }
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
