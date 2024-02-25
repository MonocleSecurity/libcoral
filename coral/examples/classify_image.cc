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

#include <iostream>//TODO

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
  LIB_CORAL_CONTEXT(std::unique_ptr<tflite::FlatBufferModel>&& flatbuffermodel)
    : flatbuffermodel_(std::move(flatbuffermodel))
  {
  }

//TODO do we need a mutex? probably not?
  std::unique_ptr<tflite::FlatBufferModel> flatbuffermodel_;
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
//TODO perhaps a std::map<LIB_CORAL_DEVICE_OUTPUT*, LIB_CORAL_DEVICE_OUTPUT_DATA>?
//TODO then a user can use their LIB_CORAL_DEVICE to get a pointer to the output data somehow?

};

struct BBox
{
//TODO constructor

//TODO rename please
  float ymin;
  float xmin;
  float ymax;
  float xmax;

};

struct Object
{
//TODO constructor

//TODO rename
  int id;
  float score;
  BBox bbox;
};

struct ObjectComparator
{
  bool operator()(const Object& lhs, const Object& rhs) const
  {
    return std::tie(lhs.score, lhs.id) > std::tie(rhs.score, rhs.id);
  }
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

std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* edgetpu_context)//TODO
{
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk)
  {
    return nullptr;
  }

//TODO are we sure it's going on the edge tpu unless we do this?
//TODO  auto* delegate = edgetpu_create_delegate(static_cast<edgetpu_device_type>(edgetpu_context->GetDeviceEnumRecord().type), edgetpu_context->GetDeviceEnumRecord().path.c_str(), nullptr, 0);//TODO delete somewhere?
//TODO  interpreter->ModifyGraphWithDelegate({ delegate, edgetpu_free_delegate });
//TODO  interpreter->ModifyGraphWithDelegate(delegate);

  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
  if (interpreter->SetNumThreads(1) != kTfLiteOk)
  {
    return nullptr;
  }
  if (interpreter->AllocateTensors() != kTfLiteOk)
  {
    return nullptr;
  }
  return interpreter;
}

std::vector<Object> GetDetectionResults(absl::Span<const float> bboxes, absl::Span<const float> ids, absl::Span<const float> scores, size_t count, float threshold, size_t top_k)
{
  std::priority_queue<Object, std::vector<Object>, ObjectComparator> q;
  for (size_t i = 0; i < count; ++i)
  {
    const int id = std::round(ids[i]);
    const float score = scores[i];
    if (score < threshold) continue;
    const float ymin = std::max(0.0f, bboxes[4 * i]);
    const float xmin = std::max(0.0f, bboxes[4 * i + 1]);
    const float ymax = std::min(1.0f, bboxes[4 * i + 2]);
    const float xmax = std::min(1.0f, bboxes[4 * i + 3]);
    q.push(Object{id, score, BBox{ymin, xmin, ymax, xmax}});//TODO call constructor please
    if (q.size() > top_k) q.pop();
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

//TODO
std::vector<Object> GetDetectionResults(const tflite::Interpreter& interpreter, float threshold = -std::numeric_limits<float>::infinity(), size_t top_k = std::numeric_limits<size_t>::max())
{
  absl::Span<const float> bboxes, ids, scores, count;
  // If a model has signature, we use the signature output tensor names to parse
  // the results. Otherwise, we parse the results based on some assumption of
  // the output tensor order and size.
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
  return GetDetectionResults(bboxes, ids, scores, static_cast<size_t>(count[0]), threshold, top_k);
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
  LIB_CORAL_CONTEXT* coralcontext = new LIB_CORAL_CONTEXT(std::move(flatbuffermodel));
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

LIB_CORAL_MODULE_API LIB_CORAL_DEVICE* LibCoralOpenDevice(LIB_CORAL_CONTEXT* context, const size_t index)
{
  // Open device
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpucontext = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice((context->records_[index].type_ == LIB_CORAL_DEVICE_TYPE::PCI) ? edgetpu::DeviceType::kApexPci : edgetpu::DeviceType::kApexUsb, LibCoralGetDevicePath(context, index));
  if (edgetpucontext == nullptr)
  {
    return nullptr;
  }
  // Build model
  std::unique_ptr<tflite::Interpreter> interpreter = std::move(BuildEdgeTpuInterpreter(*context->flatbuffermodel_, edgetpucontext.get()));
  if (interpreter == nullptr)
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

  interpreter->AllocateTensors();//TODO

  return new LIB_CORAL_DEVICE(edgetpucontext, std::move(interpreter), std::array<int, 3>({ inputdims->data[1], inputdims->data[2], inputdims->data[3] }), std::array<int, 3>({ outputdims->data[1], outputdims->data[2], outputdims->data[3] }));
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
  std::lock_guard<std::mutex> lock(device->mutex_);
  uint8_t* input = device->interpreter_->typed_input_tensor<uint8_t>(0);
  std::memcpy(input, data, size);
  const TfLiteStatus status = device->interpreter_->Invoke();
  if (status != TfLiteStatus::kTfLiteOk)
  {
    return 1;
  }
  // Collect results
//TODO I think we collect all things and package it in a way that can be ready nicely
  const std::vector<Object> objects = GetDetectionResults(*device->interpreter_);
  for (auto o : objects)
  {
    if (o.id == 0 || o.id == 1)
    {
      if (o.score > 0.7)
      {
        std::cout << o.id << " " << o.score << " " << o.bbox.xmin << " " << o.bbox.ymin << std::endl;//TODO
      }
    }
  }
  return 0;
}

LIB_CORAL_MODULE_API const float* const LibCoralGetResults(LIB_CORAL_DEVICE* device)//TODO changes now... returns Object* I think?
{
  return nullptr;//TODO
//TODO  return device->resultsbuffer_.data();
}

LIB_CORAL_MODULE_API size_t LibCoralGetResultsSize(LIB_CORAL_DEVICE* device)//TODO returns number of objects I think?
{
  return 0;//TODO
//TODO  return device->resultsbuffer_.size();
}

}

