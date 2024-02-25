// coraldevice.cpp
//

///// Includes /////

#include <algorithm>
#include <array>
#include <fstream>
#include <mutex>
#include <numeric>
#include <queue>
#include <regex>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/substitute.h"
#include "coral/classification/adapter.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "tensorflow/lite/interpreter.h"
#include "coraldevice.h"

///// Declarations /////

struct LIB_CORAL_DEVICE;

///// Structures /////

struct Object
{
  Object(const int id, const float score, const std::array<float, 4>& boundingbox)
    : id_(id)
    , score_(score)
    , boundingbox_(boundingbox)
  {
  }

  int id_;
  float score_;
  std::array<float, 4> boundingbox_; // ymin, xmin, ymax, xmax
};

struct ObjectComparator
{
  bool operator()(const Object& lhs, const Object& rhs) const
  {
    return std::tie(lhs.score_, lhs.id_) > std::tie(rhs.score_, rhs.id_);
  }
};

struct LIB_CORAL_CONTEXT_RECORD
{
  LIB_CORAL_CONTEXT_RECORD(const LIB_CORAL_DEVICE_TYPE type, const std::string& path)
    : type_(type)
    , path_(path)
    , device_(nullptr)
  {
  }

  const LIB_CORAL_DEVICE_TYPE type_;
  const std::string path_;
  LIB_CORAL_DEVICE* device_;

};

struct LIB_CORAL_CONTEXT
{
  LIB_CORAL_CONTEXT(std::unique_ptr<tflite::FlatBufferModel>&& flatbuffermodel, const std::vector<LIB_CORAL_CONTEXT_RECORD>& records)
    : flatbuffermodel_(std::move(flatbuffermodel))
    , records_(records)
  {
  }

  const std::unique_ptr<tflite::FlatBufferModel> flatbuffermodel_;
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

  mutable std::mutex mutex_;
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpucontext_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::array<int, 3> inputshape_;
  std::array<int, 3> outputshape_;

};

struct LIB_CORAL_DEVICE_CONTAINER
{
  LIB_CORAL_DEVICE_CONTAINER(LIB_CORAL_DEVICE* device)
    : device_(device)
  {
  }

  LIB_CORAL_DEVICE* device_;
  std::vector<Object> results_;

};

///// Functions /////

std::vector<int> TensorShape(const TfLiteTensor& tensor)
{
  return std::vector<int>(tensor.dims->data, tensor.dims->data + tensor.dims->size);
}

absl::Span<const float> TensorData(const TfLiteTensor& tensor)
{
  return absl::MakeSpan(reinterpret_cast<const float*>(tensor.data.data), tensor.bytes / sizeof(float));
}

TfLiteFloatArray* TfLiteFloatArrayCopy(const TfLiteFloatArray* src)
{
  if (!src)
  {
    return nullptr;
  }
  TfLiteFloatArray* copy = static_cast<TfLiteFloatArray*>(malloc(TfLiteFloatArrayGetSizeInBytes(src->size)));
  copy->size = src->size;
  std::memcpy(copy->data, src->data, src->size * sizeof(float));
  return copy;
}

TfLiteAffineQuantization* TfLiteAffineQuantizationCopy(const TfLiteAffineQuantization* src)
{
  if (!src)
  {
    return nullptr;
  }
  TfLiteAffineQuantization* copy = static_cast<TfLiteAffineQuantization*>(malloc(sizeof(TfLiteAffineQuantization)));
  copy->scale = TfLiteFloatArrayCopy(src->scale);
  copy->zero_point = TfLiteIntArrayCopy(src->zero_point);
  copy->quantized_dimension = src->quantized_dimension;
  return copy;
}

int SetTensorBuffer(tflite::Interpreter* interpreter, int tensor_index, const void* buffer, size_t buffer_size)
{
  const auto* tensor = interpreter->tensor(tensor_index);
  auto quantization = tensor->quantization;
  if (quantization.type != kTfLiteNoQuantization)
  {
    // Deep copy quantization parameters.
    if (quantization.type != kTfLiteAffineQuantization)
    {
      return 2;
    }
    quantization.params = TfLiteAffineQuantizationCopy(reinterpret_cast<TfLiteAffineQuantization*>(quantization.params));
  }
  const std::vector<int> shape = TensorShape(*tensor);
  if (interpreter->SetTensorParametersReadOnly(tensor_index, tensor->type, tensor->name, std::vector<int>(shape.begin(), shape.end()), quantization, reinterpret_cast<const char*>(buffer), buffer_size) != kTfLiteOk)
  {
    return 3;
  }
  return 0;
}

std::vector<Object> GetDetectionResults(const absl::Span<const float> bboxes, const absl::Span<const float> ids, const absl::Span<const float> scores, const size_t count, const float threshold, const size_t topk)
{
  std::priority_queue<Object, std::vector<Object>, ObjectComparator> q;
  for (size_t i = 0; i < count; ++i)
  {
    const int id = std::round(ids[i]);
    const float score = scores[i];
    if (score < threshold)
    {
      continue;
    }
    const float ymin = std::max(0.0f, bboxes[4 * i]);
    const float xmin = std::max(0.0f, bboxes[4 * i + 1]);
    const float ymax = std::min(1.0f, bboxes[4 * i + 2]);
    const float xmax = std::min(1.0f, bboxes[4 * i + 3]);
    q.push(Object(id, score, {ymin, xmin, ymax, xmax}));
    if (q.size() > topk) q.pop();
  }

  std::vector<Object> ret;
  ret.reserve(q.size());
  while (!q.empty())
  {
    ret.push_back(q.top());
    q.pop();
  }
  std::reverse(ret.begin(), ret.end());
  return ret;
}

std::vector<Object> GetDetectionResults(const tflite::Interpreter& interpreter, const float threshold, const size_t topk)
{
  absl::Span<const float> bboxes;
  absl::Span<const float> ids;
  absl::Span<const float> scores;
  absl::Span<const float> count;
  if (!interpreter.signature_def_names().empty())
  {
    const auto& signature_output_map = interpreter.signature_outputs(interpreter.signature_def_names()[0]->c_str());
    count = TensorData(*interpreter.tensor(signature_output_map.at("output_0")));
    scores = TensorData(*interpreter.tensor(signature_output_map.at("output_1")));
    ids = TensorData(*interpreter.tensor(signature_output_map.at("output_2")));
    bboxes = TensorData(*interpreter.tensor(signature_output_map.at("output_3")));
  }
  else if (interpreter.output_tensor(3)->bytes / sizeof(float) == 1)
  {
    bboxes = TensorData(*interpreter.output_tensor(0));
    ids = TensorData(*interpreter.output_tensor(1));
    scores = TensorData(*interpreter.output_tensor(2));
    count = TensorData(*interpreter.output_tensor(3));
  }
  else
  {
    scores = TensorData(*interpreter.output_tensor(0));
    bboxes = TensorData(*interpreter.output_tensor(1));
    count = TensorData(*interpreter.output_tensor(2));
    ids = TensorData(*interpreter.output_tensor(3));
  }
  return GetDetectionResults(bboxes, ids, scores, static_cast<size_t>(count[0]), threshold, topk);
}

extern "C"
{

LIB_CORAL_MODULE_API LIB_CORAL_CONTEXT* LibCoralInit(const char* model)
{
  // Load model from disk
  std::unique_ptr<tflite::FlatBufferModel> flatbuffermodel = tflite::FlatBufferModel::BuildFromFile(model);
  if (flatbuffermodel == nullptr)
  {
    return nullptr;
  }
  // Enumerate devices
  const std::vector<edgetpu::EdgeTpuManager::DeviceEnumerationRecord> records = edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
  std::vector<LIB_CORAL_CONTEXT_RECORD> result;
  for (const edgetpu::EdgeTpuManager::DeviceEnumerationRecord& record : records)
  {
    if (record.type == edgetpu::DeviceType::kApexUsb)
    {
      result.push_back(LIB_CORAL_CONTEXT_RECORD(LIB_CORAL_DEVICE_TYPE::USB, record.path));
    }
    else if (record.type == edgetpu::DeviceType::kApexPci)
    {
      result.push_back(LIB_CORAL_CONTEXT_RECORD(LIB_CORAL_DEVICE_TYPE::PCI, record.path));
    }
  }
  return new LIB_CORAL_CONTEXT(std::move(flatbuffermodel), result);
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

LIB_CORAL_MODULE_API LIB_CORAL_DEVICE_CONTAINER* LibCoralOpenDevice(LIB_CORAL_CONTEXT* context, const size_t index)
{
  // If the device is already open, then just return that
  if (context->records_[index].device_)
  {
    return new LIB_CORAL_DEVICE_CONTAINER(context->records_[index].device_);
  }
  // Open device
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpucontext = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice((context->records_[index].type_ == LIB_CORAL_DEVICE_TYPE::PCI) ? edgetpu::DeviceType::kApexPci : edgetpu::DeviceType::kApexUsb, LibCoralGetDevicePath(context, index));
  if (edgetpucontext == nullptr)
  {
    return nullptr;
  }
  // Build model
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(*context->flatbuffermodel_, resolver)(&interpreter) != kTfLiteOk)
  {
    return nullptr;
  }

//TODO are we sure it's going on the edge tpu unless we do this?
//TODO  auto* delegate = edgetpu_create_delegate(static_cast<edgetpu_device_type>(edgetpu_context->GetDeviceEnumRecord().type), edgetpu_context->GetDeviceEnumRecord().path.c_str(), nullptr, 0);//TODO delete somewhere?
//TODO  interpreter->ModifyGraphWithDelegate({ delegate, edgetpu_free_delegate });
//TODO  interpreter->ModifyGraphWithDelegate(delegate);

  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpucontext.get());
  if (interpreter->SetNumThreads(1) != kTfLiteOk)
  {
    return nullptr;
  }
  if (interpreter->AllocateTensors() != kTfLiteOk)
  {
    return nullptr;
  }
  // Retrieve the input shape
  if (interpreter->inputs().size() == 0)
  {
    return nullptr;
  }
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
  if (interpreter->AllocateTensors() != kTfLiteOk)
  {
    return nullptr;
  }
  LIB_CORAL_DEVICE* device = new LIB_CORAL_DEVICE(edgetpucontext, std::move(interpreter), std::array<int, 3>({ inputdims->data[1], inputdims->data[2], inputdims->data[3] }), std::array<int, 3>({ outputdims->data[1], outputdims->data[2], outputdims->data[3] }));
  context->records_[index].device_ = device;
  return new LIB_CORAL_DEVICE_CONTAINER(device);
}

LIB_CORAL_MODULE_API void LibCoralCloseDevice(LIB_CORAL_DEVICE_CONTAINER* device)
{
  device->device_->interpreter_.reset();
  device->device_->edgetpucontext_.reset();
}

LIB_CORAL_MODULE_API const int* const LibCoralGetInputShape(LIB_CORAL_DEVICE_CONTAINER* device)
{
  return device->device_->inputshape_.data();
}

LIB_CORAL_MODULE_API const int* const LibCoralGetOutputShape(LIB_CORAL_DEVICE_CONTAINER* device)
{
  return device->device_->outputshape_.data();
}

LIB_CORAL_MODULE_API int LibCoralRun(LIB_CORAL_DEVICE_CONTAINER* device, const uint8_t* const data, const size_t size)
{
  std::lock_guard<std::mutex> lock(device->device_->mutex_);
  uint8_t* input = device->device_->interpreter_->typed_input_tensor<uint8_t>(0);
  std::memcpy(input, data, size);
  const TfLiteStatus status = device->device_->interpreter_->Invoke();
  if (status != TfLiteStatus::kTfLiteOk)
  {
    return 1;
  }
  // Collect results
  device->results_ = GetDetectionResults(*device->device_->interpreter_, std::numeric_limits<float>::lowest(), std::numeric_limits<size_t>::max());
  return 0;
}

LIB_CORAL_MODULE_API int LibCoralGetResultId(LIB_CORAL_DEVICE_CONTAINER* device, const size_t index)
{
  return device->results_[index].id_;
}

LIB_CORAL_MODULE_API float LibCoralGetResultScore(LIB_CORAL_DEVICE_CONTAINER* device, const size_t index)
{
  return device->results_[index].score_;
}

LIB_CORAL_MODULE_API const float* const LibCoralGetResultBoundingBox(LIB_CORAL_DEVICE_CONTAINER* device, const size_t index)
{
  return device->results_[index].boundingbox_.data();
}

LIB_CORAL_MODULE_API size_t LibCoralGetResultsSize(LIB_CORAL_DEVICE_CONTAINER* device)
{
  return device->results_.size();
}

}

