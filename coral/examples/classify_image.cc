// coraldevice.cpp
//

///// Includes /////

#include <algorithm>
#include <array>
//TODO #include <edgetpu.h>
//TODO #include <edgetpu_c.h>
#include <fstream>
#include <numeric>
#include <queue>
#include <regex>
#include <vector>

#include <iostream>//TODO

//TODO retarded <>
//TODO #include "tensorflow/lite/model.h"
//TODO #include "tensorflow/lite/model_builder.h"
//TODO #include "tensorflow/lite/builtin_op_data.h"
//TODO #include "tensorflow/lite/kernels/register.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/substitute.h"
#include "coral/classification/adapter.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "tensorflow/lite/interpreter.h"
#include "coraldevice.h"

///// Structures /////

struct LIB_CORAL_CONTEXT_RECORD
{
  LIB_CORAL_CONTEXT_RECORD(const LIB_CORAL_DEVICE_TYPE type, const std::string& path)
    : type_(type)
    , path_(path)
  {
  }

  LIB_CORAL_DEVICE_TYPE type_;
  std::string path_;

};

struct LIB_CORAL_CONTEXT
{
  LIB_CORAL_CONTEXT()
  {
  }

  std::unique_ptr<tflite::FlatBufferModel> flatbuffermodel_;//TODO should probably be on the device instead? nevermind it should be reused the same one every time from here I think?
  std::vector<LIB_CORAL_CONTEXT_RECORD> records_;

};

struct LIB_CORAL_DEVICE
{
  LIB_CORAL_DEVICE(const std::shared_ptr<edgetpu::EdgeTpuContext>& edgetpucontext, std::unique_ptr<tflite::Interpreter>&& interpreter, const std::array<int, 3>& inputshape, const std::array<int, 3>& outputshape)
    : edgetpucontext_(edgetpucontext)
    , interpreter_(std::move(interpreter))
    , inputshape_(inputshape)
    , outputshape_(outputshape)
  {
  }

  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpucontext_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::array<int, 3> inputshape_;
  std::array<int, 3> outputshape_;
  std::vector<uint8_t> inputbuffer_;
  std::vector<float> resultsbuffer_;

};

///// Functions /////

std::vector<int> TensorShape(const TfLiteTensor& tensor)
{
  return std::vector<int>(tensor.dims->data, tensor.dims->data + tensor.dims->size);
}

template <typename T>
absl::Span<const T> TensorData(const TfLiteTensor& tensor) {
  return absl::MakeSpan(reinterpret_cast<const T*>(tensor.data.data),
                        tensor.bytes / sizeof(T));
}

// Gets the mutable data from the given tensor.
template <typename T>
absl::Span<T> MutableTensorData(const TfLiteTensor& tensor) {
  return absl::MakeSpan(reinterpret_cast<T*>(tensor.data.data),
                        tensor.bytes / sizeof(T));
}

int TensorSize(const TfLiteTensor& tensor)
{
  auto shape = TensorShape(tensor);
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

template <typename InputIt, typename OutputIt>
OutputIt Dequantize(InputIt first, InputIt last, OutputIt d_first, float scale,
                    int32_t zero_point) {
  while (first != last) *d_first++ = scale * (*first++ - zero_point);
  return d_first;
}

// Returns a dequantized version of the given vector span.
template <typename T, typename OutputIt>
OutputIt Dequantize(absl::Span<const T> span, OutputIt d_first, float scale,
                    int32_t zero_point) {
  return Dequantize(span.begin(), span.end(), d_first, scale, zero_point);
}

template <typename T>
std::vector<T> DequantizeTensor(const TfLiteTensor& tensor) {
  const auto scale = tensor.params.scale;
  const auto zero_point = tensor.params.zero_point;
  std::vector<T> result(TensorSize(tensor));

  if (tensor.type == kTfLiteUInt8)
    Dequantize(TensorData<uint8_t>(tensor), result.begin(), scale, zero_point);
  else if (tensor.type == kTfLiteInt8)
    Dequantize(TensorData<int8_t>(tensor), result.begin(), scale, zero_point);
  else
    LOG(FATAL) << "Unsupported tensor type: " << tensor.type;

  return result;
}

struct Class {
  // The class label id.
  int id;
  // The prediction score.
  float score;
};

struct ClassComparator {
  bool operator()(const Class& lhs, const Class& rhs) const {
    return std::tie(lhs.score, lhs.id) > std::tie(rhs.score, rhs.id);
  }
};

std::string ToString(const Class& c) {
  return absl::Substitute("Class(id=$0,score=$1)", c.id, c.score);
}

std::vector<Class> GetClassificationResults(absl::Span<const float> scores,
                                            float threshold, size_t top_k) {
  std::priority_queue<Class, std::vector<Class>, ClassComparator> q;
  for (int i = 0; i < scores.size(); ++i) {
    if (scores[i] < threshold) continue;
    q.push(Class{i, scores[i]});
    if (q.size() > top_k) q.pop();
  }

  std::vector<Class> ret;
  while (!q.empty()) {
    ret.push_back(q.top());
    q.pop();
  }
  std::reverse(ret.begin(), ret.end());
  return ret;
}

std::vector<Class> GetClassificationResults(
    const tflite::Interpreter& interpreter, float threshold, size_t top_k) {
  const auto& tensor = *interpreter.output_tensor(0);
  if (tensor.type == kTfLiteUInt8 || tensor.type == kTfLiteInt8) {
    return GetClassificationResults(DequantizeTensor<float>(tensor), threshold,
                                    top_k);
  } else if (tensor.type == kTfLiteFloat32) {
    return GetClassificationResults(TensorData<float>(tensor), threshold,
                                    top_k);
  } else {
    LOG(FATAL) << "Unsupported tensor type: " << tensor.type;
  }
}

TfLiteFloatArray* TfLiteFloatArrayCopy(const TfLiteFloatArray* src) {
  if (!src) return nullptr;

  auto* copy = static_cast<TfLiteFloatArray*>(
    malloc(TfLiteFloatArrayGetSizeInBytes(src->size)));
  copy->size = src->size;
  std::memcpy(copy->data, src->data, src->size * sizeof(float));
  return copy;
}

TfLiteAffineQuantization* TfLiteAffineQuantizationCopy(
  const TfLiteAffineQuantization* src) {
  if (!src) return nullptr;

  auto* copy = static_cast<TfLiteAffineQuantization*>(
    malloc(sizeof(TfLiteAffineQuantization)));
  copy->scale = TfLiteFloatArrayCopy(src->scale);
  copy->zero_point = TfLiteIntArrayCopy(src->zero_point);
  copy->quantized_dimension = src->quantized_dimension;
  return copy;
}
//TODO
int SetTensorBuffer(tflite::Interpreter* interpreter, int tensor_index, const void* buffer, size_t buffer_size)
{
  const auto* tensor = interpreter->tensor(tensor_index);

  auto quantization = tensor->quantization;
  if (quantization.type != kTfLiteNoQuantization) {
    // Deep copy quantization parameters.
    if (quantization.type != kTfLiteAffineQuantization)
      return 2;
    quantization.params = TfLiteAffineQuantizationCopy(
      reinterpret_cast<TfLiteAffineQuantization*>(quantization.params));
  }

  const auto shape = TensorShape(*tensor);
  if (interpreter->SetTensorParametersReadOnly(
    tensor_index, tensor->type, tensor->name,
    std::vector<int>(shape.begin(), shape.end()), quantization,
    reinterpret_cast<const char*>(buffer), buffer_size) != kTfLiteOk)
  {
    return 3;
  }
  return 0;
}


std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* edgetpu_context)//TODO
{
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk)
  {
    std::cerr << "Failed to build interpreter." << std::endl;
  }

//TODO  auto* delegate = edgetpu_create_delegate(static_cast<edgetpu_device_type>(edgetpu_context->GetDeviceEnumRecord().type), edgetpu_context->GetDeviceEnumRecord().path.c_str(), nullptr, 0);//TODO delete somewhere?
//TODO  interpreter->ModifyGraphWithDelegate({ delegate, edgetpu_free_delegate });
//TODO  interpreter->ModifyGraphWithDelegate(delegate);

  // Bind given context with interpreter.
  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
  //TODO if (interpreter->SetNumThreads(1) != kTfLiteOk)
  {
    //TODO std::cerr << "bla." << std::endl;//TODO
  }
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
  }
  return interpreter;
}

int CheckInputSize(const TfLiteTensor& tensor, size_t size)
{
  const size_t tensor_size = TensorSize(tensor);
  if (size < tensor_size)
    return 1;
  return 0;
}

std::pair<int, std::vector<float>> RunInference(const std::vector<uint8_t>& input_data, tflite::Interpreter* interpreter)//TODO tidy up
{
  std::cout << "running " << input_data.size() << " " << interpreter->inputs().size() << " " << interpreter->tensor(interpreter->inputs()[0])->dims->data[0] << " " << interpreter->tensor(interpreter->inputs()[0])->dims->data[1] << " " << interpreter->tensor(interpreter->inputs()[0])->dims->data[2] << " " << interpreter->tensor(interpreter->inputs()[0])->dims->data[3] << std::endl;//TODO

//TODO  const int input_tensor_index = interpreter->inputs()[0];
//TODO  TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_index);

//TODO  int sizey = TensorSize(*input_tensor);
//TODO  int* buffer = new int[sizey * 100];

//TODO  std::cout << "SET TENSOR BUFFER " << sizey << " " << SetTensorBuffer(interpreter, input_tensor_index, buffer, input_tensor->bytes) << std::endl;//TODO


  std::vector<float> output_data;
  uint8_t* input = interpreter->typed_input_tensor<uint8_t>(0);
  std::memcpy(input, input_data.data(), input_data.size());

  std::cout << "invoking " << interpreter << " " << input_data.size() << std::endl;//TODO

  const TfLiteStatus status = interpreter->Invoke();
  if (status != TfLiteStatus::kTfLiteOk)
  {
  std::cout << "failinvoke " << interpreter << " " << input_data.size() << std::endl;//TODO

  }

//TODO is this correct? no, we want object detector results, not this shit
  for (auto result : coral::GetClassificationResults(*interpreter, 0.0f, /*top_k=*/3)) {
    std::cout << "---------------------------" << std::endl;
    std::cout << "Score: " << result.score << std::endl;
  }



  //TODO parse output
  const auto& output_indices = interpreter->outputs();
  const int num_outputs = output_indices.size();
  int out_idx = 0;
  for (int i = 0; i < num_outputs; ++i)
  {
    const auto* out_tensor = interpreter->tensor(output_indices[i]);
    assert(out_tensor != nullptr);
    if (out_tensor->type == kTfLiteUInt8)
    {
      const int num_values = out_tensor->bytes;
      output_data.resize(out_idx + num_values);
      const uint8_t* output = interpreter->typed_output_tensor<uint8_t>(i);
      for (int j = 0; j < num_values; ++j)
      {
        output_data[out_idx++] = (output[j] - out_tensor->params.zero_point) * out_tensor->params.scale;
	std::cout << output_data[out_idx - 1] << std::endl;//TODO
      }
    }
    else if (out_tensor->type == kTfLiteFloat32)//TODO never true I think
    {
      const int num_values = out_tensor->bytes / sizeof(float);
      output_data.resize(out_idx + num_values);
      const float* output = interpreter->typed_output_tensor<float>(i);
      for (int j = 0; j < num_values; ++j)
      {
        output_data[out_idx++] = output[j];
//TODO	std::cout << output_data[out_idx - 1] << std::endl;//TODO
      }
    }
    else
    {
      std::cerr << "Tensor " << out_tensor->name << " has unsupported output type: " << out_tensor->type << std::endl;
    }
  }
  std::cout << "finished " << interpreter << " " << input_data.size() << " " << output_data.size() << std::endl;//TODO
  return std::make_pair(0, output_data);
}

extern "C"
{

LIB_CORAL_MODULE_API LIB_CORAL_CONTEXT* LibCoralInit()
{
  const std::vector<edgetpu::EdgeTpuManager::DeviceEnumerationRecord> records = edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
  LIB_CORAL_CONTEXT* coralcontext = new LIB_CORAL_CONTEXT();
  for (const edgetpu::EdgeTpuManager::DeviceEnumerationRecord& record : records)
  {
    if (record.type == edgetpu::DeviceType::kApexUsb)
    {
      coralcontext->records_.push_back(LIB_CORAL_CONTEXT_RECORD(LIB_CORAL_DEVICE_TYPE::USB, record.path));
    }
    else if (record.type == edgetpu::DeviceType::kApexPci)
    {
      coralcontext->records_.push_back(LIB_CORAL_CONTEXT_RECORD(LIB_CORAL_DEVICE_TYPE::PCI, record.path));
    }
  }
  return coralcontext;
}

LIB_CORAL_MODULE_API void LibCoralDestroy(LIB_CORAL_CONTEXT* context)
{
  if (context)
  {
    delete context;
  }
}

LIB_CORAL_MODULE_API size_t LibCoralGetNumDevices(LIB_CORAL_CONTEXT* context)
{
  return context->records_.size();
}

LIB_CORAL_MODULE_API LIB_CORAL_DEVICE_TYPE LibCoralGetDeviceType(LIB_CORAL_CONTEXT* context, const size_t index)
{
  return context->records_[index].type_;
}

LIB_CORAL_MODULE_API const char* LibCoralGetDevicePath(LIB_CORAL_CONTEXT* context, const size_t index)
{
  return context->records_[index].path_.c_str();
}

LIB_CORAL_MODULE_API LIB_CORAL_DEVICE* LibCoralOpenDevice(LIB_CORAL_CONTEXT* context, const size_t index, const char* model)
{
  std::cout << "LibCoralOpenDevice 1" << std::endl;//TODO
  // Load model from disk
  context->flatbuffermodel_ = tflite::FlatBufferModel::BuildFromFile(model);
  if (context->flatbuffermodel_ == nullptr)
  {
    return nullptr;
  }
  std::cout << "LibCoralOpenDevice 2 " << (int)((context->records_[index].type_ == LIB_CORAL_DEVICE_TYPE::PCI) ? edgetpu::DeviceType::kApexPci : edgetpu::DeviceType::kApexUsb) << " " << LibCoralGetDevicePath(context, index) << std::endl;//TODO
  // Open device
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpucontext = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  //TODO std::shared_ptr<edgetpu::EdgeTpuContext> edgetpucontext = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice((context->records_[index].type_ == LIB_CORAL_DEVICE_TYPE::PCI) ? edgetpu::DeviceType::kApexPci : edgetpu::DeviceType::kApexUsb, LibCoralGetDevicePath(context, index));
  if (edgetpucontext == nullptr)
  {
    return nullptr;
  }
  std::cout << "LibCoralOpenDevice 3" << std::endl;//TODO
  // Build model
  std::unique_ptr<tflite::Interpreter> interpreter = std::move(BuildEdgeTpuInterpreter(*context->flatbuffermodel_, edgetpucontext.get()));//TODO pass in normal... store this value too?
  if (interpreter == nullptr)
  {
    return nullptr;
  }
  std::cout << "LibCoralOpenDevice 4 " << ((int)interpreter->tensor(interpreter->inputs()[0])->type) << std::endl;//TODO
  // Retrieve the input shape
  if (interpreter->inputs().size() == 0)
  {
    return nullptr;
  }
  std::cout << "LibCoralOpenDevice 6" << std::endl;//TODO
  const TfLiteIntArray* inputdims = interpreter->tensor(interpreter->inputs()[0])->dims;
  if (inputdims == nullptr)
  {
    return nullptr;
  }
  // Retrieve the output shape
  if (interpreter->outputs().size() == 0)
  {
    return nullptr;
  }
  const TfLiteIntArray* outputdims = interpreter->tensor(interpreter->outputs()[0])->dims;
  if (outputdims == nullptr)
  {
    return nullptr;
  }

  interpreter->AllocateTensors();//TODO

  std::cout << "LibCoralOpenDevice 7 " << interpreter->outputs().size() << " " << interpreter->inputs().size() << " " << inputdims->data[0] << " " << outputdims->data[0] << std::endl;//TODO
  return new LIB_CORAL_DEVICE(edgetpucontext, std::move(interpreter), std::array<int, 3>({ inputdims->data[1], inputdims->data[2], inputdims->data[3] }), std::array<int, 3>({ outputdims->data[1], outputdims->data[2], outputdims->data[3] }));//TODO 1,2,3???
}

LIB_CORAL_MODULE_API void LibCoralCloseDevice(LIB_CORAL_DEVICE* device)
{
  device->interpreter_.reset();
  device->edgetpucontext_.reset();
}

LIB_CORAL_MODULE_API const int* const LibCoralGetInputShape(LIB_CORAL_DEVICE* device)
{
  return device->inputshape_.data();
}

LIB_CORAL_MODULE_API const int* const LibCoralGetOutputShape(LIB_CORAL_DEVICE* device)
{
  return device->outputshape_.data();
}

LIB_CORAL_MODULE_API int LibCoralRun(LIB_CORAL_DEVICE* device, const uint8_t* const data, const size_t size)
{
  device->inputbuffer_.clear();
  device->inputbuffer_.insert(device->inputbuffer_.begin(), data, data + size);//TODO I don't think we need this buffer
  const std::pair<int, std::vector<float>> result = RunInference(device->inputbuffer_, device->interpreter_.get());
  if (result.first)
  {
    return result.first;
  }
  device->resultsbuffer_ = result.second;
  return 0;
}

LIB_CORAL_MODULE_API const float* const LibCoralGetResults(LIB_CORAL_DEVICE* device)
{
  return device->resultsbuffer_.data();
}

LIB_CORAL_MODULE_API size_t LibCoralGetResultsSize(LIB_CORAL_DEVICE* device)
{
  return device->resultsbuffer_.size();
}

}

