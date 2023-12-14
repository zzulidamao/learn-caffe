#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
//Blob的最大维度32，默认4维，NCHW
const int kMaxBlobAxes = 32;

namespace caffe {
//在caffe命名空间下
/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *Blob就是封装了同步内存数据的类，该类作为计算单元与层，网络，和优化器进行基本的计算交互
 * TODO(dox): more thorough description.
 */
//类模板
template <typename Dtype>
class Blob {
 public:
  Blob()
       : data_(), diff_(), count_(0), capacity_(0) {}//默认构造函数：初始化列表
  
  /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
  explicit Blob(const int num, const int channels, const int height,
      const int width);//不能使用=赋值给新对象，只能严格定义对象，带参构造函数n,c,h,w（荒废了）
  explicit Blob(const vector<int>& shape);//通过vector<int>进行初始化，推荐使用

  /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
  void Reshape(const int num, const int channels, const int height,
      const int width);//把Blob按指定形状进行reshape（荒废了）
  /**
   * @brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *改变Blob的维度，申请内存。reshape这个函数是申请内存并进行初始化，以及改变输出blob的维度在reshape或者前向传播的时候。
   *更改blob的尺寸的时候，只有内存不够用的情况下会重新分配内存，并且多余的内存不会被释放
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *blob进行reshape之后立刻调用backward是错误的，要在调用了forward或者reshape进行前向传播到更高的层，才能调用backward
   * Note that reshaping an input blob and immediately calling Net::Backward is
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   */
  void Reshape(const vector<int>& shape);//推荐使用
  void Reshape(const BlobShape& shape);
  void ReshapeLike(const Blob& other);//使一个blob变成相同的形状
  inline string shape_string() const {
    ostringstream stream;
    for (int i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";//N C H W 
    }
    stream << "(" << count_ << ")";
    return stream.str();//返回形状数据组成的字符串
  }
  inline const vector<int>& shape() const { return shape_; }
  /**
   * @brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   *返回索引的维度，索引为-1就从最后才是索引
   * @param index the axis index, which may be negative as it will be
   *        "canonicalized" using CanonicalAxisIndex.
   *        Dies on out of range index.//超出索引挂掉
   */
  inline int shape(int index) const {
    return shape_[CanonicalAxisIndex(index)];
  }
  inline int num_axes() const { return shape_.size(); }
  inline int count() const { return count_; }

  /**
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *计算切片的体积
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
  inline int count(int start_axis, int end_axis) const {
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
      count *= shape(i);
    }
    return count;
  }
  /**
   * @brief Compute the volume of a slice spanning from a particular first
   *        axis to the final axis.
   *计算特定轴开始到最后轴切片的体积
   * @param start_axis The first axis to include in the slice.
   */
  inline int count(int start_axis) const {
    return count(start_axis, num_axes());
  }

  /**
   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *返回用户指定轴的规范版本
   * @param axis_index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
  inline int CanonicalAxisIndex(int axis_index) const {
    CHECK_GE(axis_index, -num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    CHECK_LT(axis_index, num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes();
    }
    return axis_index;
  }

  /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
  inline int num() const { return LegacyShape(0); }//废弃
  /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
  inline int channels() const { return LegacyShape(1); }//废弃
  /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
  inline int height() const { return LegacyShape(2); }//废弃
  /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
  inline int width() const { return LegacyShape(3); }//废弃
  inline int LegacyShape(int index) const {
    CHECK_LE(num_axes(), 4)
        << "Cannot use legacy accessors on Blobs with > 4 axes.";
    CHECK_LT(index, 4);
    CHECK_GE(index, -4);
    if (index >= num_axes() || index < -num_axes()) {
      // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
      // indexing) -- this special case simulates the one-padding used to fill
      // extraneous axes of legacy blobs.
      return 1;
    }
    return shape(index);
  }

  inline int offset(const int n, const int c = 0, const int h = 0,
      const int w = 0) const {
    CHECK_GE(n, 0);
    CHECK_LE(n, num());
    CHECK_GE(channels(), 0);
    CHECK_LE(c, channels());
    CHECK_GE(height(), 0);
    CHECK_LE(h, height());
    CHECK_GE(width(), 0);
    CHECK_LE(w, width());
    return ((n * channels() + c) * height() + h) * width() + w;//？
  }

  inline int offset(const vector<int>& indices) const {
    CHECK_LE(indices.size(), num_axes());
    int offset = 0;
    for (int i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        CHECK_GE(indices[i], 0);
        CHECK_LT(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }
  /**
   * @brief Copy from a source Blob.
   *从一个blob拷贝数据，第二个参数为false就拷贝data为true就拷贝diff
   * @param source the Blob to copy from
   * @param copy_diff if false, copy the data; if true, copy the diff
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   */
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);

  inline Dtype data_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_data()[offset(n, c, h, w)];
  }

  inline Dtype diff_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_diff()[offset(n, c, h, w)];
  }

  inline Dtype data_at(const vector<int>& index) const {
    return cpu_data()[offset(index)];
  }

  inline Dtype diff_at(const vector<int>& index) const {
    return cpu_diff()[offset(index)];
  }

  inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;
  }

  inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;
  }

  const Dtype* cpu_data() const;
  void set_cpu_data(Dtype* data);
  const int* gpu_shape() const;
  const Dtype* gpu_data() const;
  void set_gpu_data(Dtype* data);
  const Dtype* cpu_diff() const;
  const Dtype* gpu_diff() const;
  Dtype* mutable_cpu_data();
  Dtype* mutable_gpu_data();
  Dtype* mutable_cpu_diff();
  Dtype* mutable_gpu_diff();
  void Update();
  void FromProto(const BlobProto& proto, bool reshape = true);//从proto文件中反序列化blob
  void ToProto(BlobProto* proto, bool write_diff = false) const;//把blob序列化到proto

  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  Dtype asum_data() const;//|x1|+|x2|
  /// @brief Compute the sum of absolute values (L1 norm) of the diff.
  Dtype asum_diff() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the data.
  Dtype sumsq_data() const;//sqrt(x1^2+x2^2)
  /// @brief Compute the sum of squares (L2 norm squared) of the diff.
  Dtype sumsq_diff() const;

  /// @brief Scale the blob data by a constant factor.
  void scale_data(Dtype scale_factor);//对data数据进行缩放
  /// @brief Scale the blob diff by a constant factor.
  void scale_diff(Dtype scale_factor);

  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *释放blob的共享内存，当共享指针被=操作时
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareData(const Blob& other);
  /**
   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareDiff(const Blob& other);

  bool ShapeEquals(const BlobProto& other);

 protected:
  shared_ptr<SyncedMemory> data_;//原始数据，共享指针，指向同步内存
  shared_ptr<SyncedMemory> diff_;//损失函数关于权重的梯度
  shared_ptr<SyncedMemory> shape_data_;//共享内存的形状NCHW智能化指针
  vector<int> shape_;//形状[1,3,224,224]NCHW
  int count_;//表示当前blob中存储的元素个数，通常与blob的shape_一起使用。形状为[1,3,224,224]的blob，count_的值为1*3*224*224=150528.用于检查blob在读取或写入的足够的元素数量
  int capacity_;//blob的最大容量
  //仅在确认需要的时候，才定义拷贝构造函数和赋值运算符；否则，请使用DISABLE_COPY_AND_ASSIGN关闭此功能。
  //我们通过拷贝构造函数和赋值运算符来实现对一个类对象的拷贝。在一些情况下，编译器会隐式的调用拷贝构造函数，比如在以值传递方式传递对象时。
  //隐式的类对象拷贝会引入很多bug和性能方面的问题，还会因为很难追踪一个类对象的行为而导致代码的可读性的降低。
  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
