// main.cpp
//

///// Includes /////

#include <array>
#include "coraldevice.h"//TODO <> please
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <string>

#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

///// Functions /////

//TODO
std::map<int, std::string> ParseLabel(const std::string& label_path)
{
  std::map<int, std::string> ret;
  std::ifstream label_file(label_path);
  if (!label_file.good()) return ret;
  for (std::string line; std::getline(label_file, line);)
  {
    std::istringstream ss(line);
    int id;
    ss >> id;
    line = std::regex_replace(line, std::regex("^ +[0-9]+ +"), "");
    ret.emplace(id, line);
  }
  return ret;
}

//TODO
std::vector<uint8_t> decode_bmp(
  const uint8_t* input, int row_size, int width, int height, int channels, bool top_down) {
std::cout << "decode_bmp 1 " << height << " " << width << " " << channels << std::endl;//TODO
  std::vector<uint8_t> output(height * width * channels);
  for (int i = 0; i < height; i++) {
    int src_pos;
    int dst_pos;

    for (int j = 0; j < width; j++) {
      if (!top_down) {
        src_pos = ((height - 1 - i) * row_size) + j * channels;
      }
      else {
        src_pos = i * row_size + j * channels;
      }

      dst_pos = (i * width + j) * channels;

      switch (channels) {
      case 1:
        output[dst_pos] = input[src_pos];
        break;
      case 3:
        // BGR -> RGB
        output[dst_pos] = input[src_pos + 2];
        output[dst_pos + 1] = input[src_pos + 1];
        output[dst_pos + 2] = input[src_pos];
        break;
      case 4:
        // BGRA -> RGBA
        output[dst_pos] = input[src_pos + 2];
        output[dst_pos + 1] = input[src_pos + 1];
        output[dst_pos + 2] = input[src_pos];
        output[dst_pos + 3] = input[src_pos + 3];
        break;
      default:
        std::cerr << "Unexpected number of channels: " << channels << std::endl;
        std::abort();
        break;
      }
    }
  }
  return output;
}

//TODO
std::vector<uint8_t> read_bmp(
  const std::string& input_bmp_name, int* width, int* height, int* channels) {
/*  int begin, end;

  std::ifstream file(input_bmp_name, std::ios::in | std::ios::binary);
  if (!file) {
    std::cerr << "input file " << input_bmp_name << " not found\n";
    std::abort();
  }

  begin = file.tellg();
  file.seekg(0, std::ios::end);
  end = file.tellg();
  size_t len = end - begin;
  std::cout << "read_bmp 1 " << len << std::endl;//TODO
  std::vector<uint8_t> img_bytes(len);
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char*>(img_bytes.data()), len);
  const int32_t header_size = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 10));
  *width = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 18));
  *height = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 22));
  const int32_t bpp = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 28));
  *channels = bpp / 8;
  *channels = 3;//TODO

  // there may be padding bytes when the width is not a multiple of 4 bytes
  // 8 * channels == bits per pixel
  const int row_size = (8 * *channels * *width + 31) / 32 * 4;

  // if height is negative, data layout is top down
  // otherwise, it's bottom up
  bool top_down = (*height < 0);

  // Decode image, allocating tensor once the image size is known
  const uint8_t* bmp_pixels = &img_bytes[header_size];
  std::cout << "read_bmp 1 " << img_bytes.size() << std::endl;//TODO
  return decode_bmp(bmp_pixels, row_size, *width, abs(*height), *channels, top_down);*/



	std::vector<uint8_t> ret;
//TODO	ret.resize(512 * 512 * 3);//TODO *10 works now
//TODO	ret.resize(4 * 320 * 320);//TODO *10 works now
	ret.resize(300 * 300 * 3);//TODO *10 works now
//TODO	ret.resize(224 * 224 * 3);//TODO *10 works now
  return ret;
}

/*TODO
std::map<int, std::string> ParseLabel(const std::string& label_path) {
  std::map<int, std::string> ret;
  std::ifstream label_file(label_path);
  if (!label_file.good()) return ret;
  for (std::string line; std::getline(label_file, line);) {
    std::istringstream ss(line);
    int id;
    ss >> id;
    line = std::regex_replace(line, std::regex("^ +[0-9]+ +"), "");
    ret.emplace(id, line);
  }
  return ret;
}*/

int main(int argc, char* argv[])
{
std::cout << "A" << std::endl;//TODO
#ifdef _WIN32
  const HMODULE module = LoadLibrary("libcoraldevice.dll");
#else
std::cout << "B" << std::endl;//TODO
  void* module = dlopen("libclassify_image.so", RTLD_LAZY | RTLD_NOW);
std::cout << "C" << std::endl;//TODO
#endif
  if (module == nullptr)
  {
#ifdef _WIN32
#ifdef NDEBUG
    const HMODULE module = LoadLibrary("Release/libcoraldevice.dll");
#else
    const HMODULE module = LoadLibrary("Debug/libcoraldevice.dll");
#endif
#endif
    if (module == nullptr)
    {
#ifdef _WIN32
      std::cout << "Failed to load library" << std::endl;
#else
      std::cout << "Failed to load library " << dlerror() << std::endl;
#endif
      return 1;
    }
  }
std::cout << "D" << std::endl;//TODO
//TODO FreeLibrary() and Destroy in right places
  // Retrieve library functions
#ifdef _WIN32
  const LIB_CORAL_INIT init = reinterpret_cast<LIB_CORAL_INIT>(GetProcAddress(module, "LibCoralInit"));
  const LIB_CORAL_DESTROY destroy = reinterpret_cast<LIB_CORAL_DESTROY>(GetProcAddress(module, "LibCoralDestroy"));
  const LIB_CORAL_GET_NUM_DEVICES getnumdevices = reinterpret_cast<LIB_CORAL_GET_NUM_DEVICES>(GetProcAddress(module, "LibCoralGetNumDevices"));
  const LIB_CORAL_GET_DEVICE_TYPE getdevicetype = reinterpret_cast<LIB_CORAL_GET_DEVICE_TYPE>(GetProcAddress(module, "LibCoralGetDeviceType"));
  const LIB_CORAL_GET_DEVICE_PATH getdevicepath = reinterpret_cast<LIB_CORAL_GET_DEVICE_PATH>(GetProcAddress(module, "LibCoralGetDevicePath"));
  const LIB_CORAL_OPEN_DEVICE opendevice = reinterpret_cast<LIB_CORAL_OPEN_DEVICE>(GetProcAddress(module, "LibCoralOpenDevice"));
  const LIB_CORAL_CLOSE_DEVICE closedevice = reinterpret_cast<LIB_CORAL_CLOSE_DEVICE>(GetProcAddress(module, "LibCoralCloseDevice"));
  const LIB_CORAL_GET_INPUT_SHAPE getinputshape = reinterpret_cast<LIB_CORAL_GET_INPUT_SHAPE>(GetProcAddress(module, "LibCoralGetInputShape"));
  const LIB_CORAL_GET_OUTPUT_SHAPE getoutputshape = reinterpret_cast<LIB_CORAL_GET_OUTPUT_SHAPE>(GetProcAddress(module, "LibCoralGetOutputShape"));
  const LIB_CORAL_RUN run = reinterpret_cast<LIB_CORAL_RUN>(GetProcAddress(module, "LibCoralRun"));
  const LIB_CORAL_GET_RESULTS getresults = reinterpret_cast<LIB_CORAL_GET_RESULTS>(GetProcAddress(module, "LibCoralGetResults"));
  const LIB_CORAL_GET_RESULTS_SIZE getresultssize = reinterpret_cast<LIB_CORAL_GET_RESULTS_SIZE>(GetProcAddress(module, "LibCoralGetResultsSize"));
#else
  const LIB_CORAL_INIT init = reinterpret_cast<LIB_CORAL_INIT>(dlsym(module, "LibCoralInit"));
  const LIB_CORAL_DESTROY destroy = reinterpret_cast<LIB_CORAL_DESTROY>(dlsym(module, "LibCoralDestroy"));
  const LIB_CORAL_GET_NUM_DEVICES getnumdevices = reinterpret_cast<LIB_CORAL_GET_NUM_DEVICES>(dlsym(module, "LibCoralGetNumDevices"));
  const LIB_CORAL_GET_DEVICE_TYPE getdevicetype = reinterpret_cast<LIB_CORAL_GET_DEVICE_TYPE>(dlsym(module, "LibCoralGetDeviceType"));
  const LIB_CORAL_GET_DEVICE_PATH getdevicepath = reinterpret_cast<LIB_CORAL_GET_DEVICE_PATH>(dlsym(module, "LibCoralGetDevicePath"));
  const LIB_CORAL_OPEN_DEVICE opendevice = reinterpret_cast<LIB_CORAL_OPEN_DEVICE>(dlsym(module, "LibCoralOpenDevice"));
  const LIB_CORAL_CLOSE_DEVICE closedevice = reinterpret_cast<LIB_CORAL_CLOSE_DEVICE>(dlsym(module, "LibCoralCloseDevice"));
  const LIB_CORAL_GET_INPUT_SHAPE getinputshape = reinterpret_cast<LIB_CORAL_GET_INPUT_SHAPE>(dlsym(module, "LibCoralGetInputShape"));
  const LIB_CORAL_GET_OUTPUT_SHAPE getoutputshape = reinterpret_cast<LIB_CORAL_GET_OUTPUT_SHAPE>(dlsym(module, "LibCoralGetOutputShape"));
  const LIB_CORAL_RUN run = reinterpret_cast<LIB_CORAL_RUN>(dlsym(module, "LibCoralRun"));
  const LIB_CORAL_GET_RESULTS getresults = reinterpret_cast<LIB_CORAL_GET_RESULTS>(dlsym(module, "LibCoralGetResults"));
  const LIB_CORAL_GET_RESULTS_SIZE getresultssize = reinterpret_cast<LIB_CORAL_GET_RESULTS_SIZE>(dlsym(module, "LibCoralGetResultsSize"));
#endif
std::cout << "E" << std::endl;//TODO
  if ((init == nullptr) || (destroy == nullptr) || (getnumdevices == nullptr) || (getdevicetype == nullptr) || (getdevicepath == nullptr) || (opendevice == nullptr) || (closedevice == nullptr) || (getinputshape == nullptr) || (getoutputshape == nullptr) || (run == nullptr) || (getresults == nullptr) || (getresultssize == nullptr))
  {
    std::cout << "Failed to load library functions" << std::endl;
    return 2;
  }
std::cout << "F" << std::endl;//TODO
  LIB_CORAL_CONTEXT* coralcontext = init();
  if (coralcontext == nullptr)
  {
    std::cout << "Failed to initialise coral library" << std::endl;
    return 3;
  }
std::cout << "F" << std::endl;//TODO
  // Enumerate devices
  const size_t numdevices = getnumdevices(coralcontext);
  std::cout << "Number of coral devices found: " << numdevices << std::endl;
  if (numdevices == 0)
  {
    //TODO
    return 4;
  }
std::cout << "G" << std::endl;//TODO
  for (size_t deviceindex = 0; deviceindex < numdevices; ++deviceindex)
  {
    const LIB_CORAL_DEVICE_TYPE type = getdevicetype(coralcontext, deviceindex);
    if (type == LIB_CORAL_DEVICE_TYPE::PCI)
    {
      std::cout << "PCI " << getdevicepath(coralcontext, deviceindex) << std::endl;
    }
    else if (type == LIB_CORAL_DEVICE_TYPE::USB)
    {
      std::cout << "USB " << getdevicepath(coralcontext, deviceindex) << std::endl;
    }
    else
    {
      std::cout << "Unknown " << getdevicepath(coralcontext, deviceindex) << std::endl;
    }
  }
  // Open device
  std::cout << "Opening device" << std::endl;
//TODO  LIB_CORAL_DEVICE* device = opendevice(coralcontext, 0, "mobilenet_v1_1.0_224_quant_edgetpu.tflite");// works
//TODO  LIB_CORAL_DEVICE* device = opendevice(coralcontext, 0, "mobilenet_v2_1.0_224_quant_edgetpu.tflite");// works

  LIB_CORAL_DEVICE* device = opendevice(coralcontext, 0, "ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite");
//TODO  LIB_CORAL_DEVICE* device = opendevice(coralcontext, 0, "efficientnet-edgetpu-S_quant_edgetpu.tflite");// works

//TODO  LIB_CORAL_DEVICE* device = opendevice(coralcontext, 0, "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite");
//TODO  LIB_CORAL_DEVICE* device = opendevice(coralcontext, 0, "efficientdet_lite3_512_ptq_edgetpu.tflite");
//TODO  LIB_CORAL_DEVICE* device = opendevice(coralcontext, 0, "ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite");
  if (device == nullptr)
  {
    std::cout << "Failed to open coral device" << std::endl;
    return 5;
  }
  // Shape
  std::cout << "Getting input shape" << std::endl;
  const int* const inputshape = getinputshape(device);
  if (inputshape == nullptr)
  {
    std::cout << "Failed to retrieve input shape" << std::endl;
    return 6;
  }
  std::cout << "INPUT SHAPE " << inputshape[0] << " " << inputshape[1] << " " << inputshape[2] << std::endl;//TODO
  const int* const outputshape = getoutputshape(device);
  if (outputshape == nullptr)
  {
    std::cout << "Failed to retrieve input shape" << std::endl;
    return 7;
  }
  std::cout << "OUTPUT SHAPE " << outputshape[0] << " " << outputshape[1] << " " << outputshape[2] << std::endl;//TODO
  // Load image
  std::cout << "Loading image" << std::endl;
  int width = 0;
  int height = 0;
  int channels = 0;
//TODO  const std::vector<uint8_t> input = read_bmp("resized_cat.bmp", &width, &height, &channels);//TODO tidy up
  const std::vector<uint8_t> input = read_bmp("output-onlinebitmaptools.bmp", &width, &height, &channels);//TODO tidy up
  std::cout << "Image loaded" << std::endl;
  if (input.empty())
  {
    std::cout << "Failed to load image" << std::endl;
    return 8;
  }
  // Inferring
  std::cout << "Running inference" << std::endl;
  const int ret = run(device, input.data(), input.size());
  if (ret)
  {
    std::cout << "Failed to run inference" << std::endl;
    return 9;
  }
  const float* const results = getresults(device);//TODO this size can be different every time...
  
  
  
  


  // Parse results
//TODO  const auto label = ParseLabel("imagenet_labels.txt");//TODO
  const auto label = ParseLabel("coco_labels.txt");//TODO
  std::cout << "8 " << label.size() << " " << outputshape[0] << " " << getresultssize(device) << std::endl;//TODO
  const size_t resultssize = getresultssize(device);
  for (size_t i = 0; i < resultssize; ++i)//TODO this size
  {
    if (results[i] != 0)
    {
      std::cout << "------\n";
//TODO      std::cout << "Class: " << label.at(i) << "\nScore: " << results[i] << "\n";
      std::cout << "Class: " << i << "\nScore: " << results[i] << "\n";
    }
  }

  //TODO




  std::cout << "yay " << ret << std::endl;//TODO
  closedevice(device);
  destroy(coralcontext);
#ifdef _WIN32
  FreeLibrary(module);
#else
  dlclose(module);
#endif
  return 0;





  //TODO remove below
  /*if (argc != 1 && argc != 4) {
    std::cout << " minimal <edgetpu model> <input resized image>" << std::endl;
    return 1;
  }
  std::cout << 1 << std::endl;//TODO
  // Modify the following accordingly to try different models and images.
  const std::string model_path = argc == 4 ? argv[1] : "mobilenet_v1_1.0_224_quant_edgetpu.tflite";
  const std::string resized_image_path = argc == 4 ? argv[2] : "resized_cat.bmp";
  const std::string label_path = argc == 4 ? argv[3] : "imagenet_labels.txt";
  std::cout << 2 << std::endl;//TODO

  // Read model.
  std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  if (model == nullptr) {
    std::cerr << "Fail to build FlatBufferModel from file: " << model_path << std::endl;
    std::abort();
  }
  std::cout << 3 << std::endl;//TODO

  for (const auto& a : edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu())
  {
    if (a.type == edgetpu::DeviceType::kApexUsb)
    {
      std::cout << "USB " << a.path << std::endl;//TODO
    }
    else if (a.type == edgetpu::DeviceType::kApexUsb)
    {
      std::cout << "USB " << a.path << std::endl;//TODO
    }
    else
    {

    }

  }

  // Build interpreter.
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context =
    edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (!edgetpu_context) {
    std::cout << "failed" << std::endl;//TODO
    //TODO interpreter = std::move(BuildInterpreter(*model));
  }
  else {
    for (const auto& a : edgetpu_context->GetDeviceOptions())
    {
      std::cout << a.first << " " << a.second << std::endl;//TODO
    }
    std::cout << "good" << std::endl;//TODO
    interpreter = std::move(BuildEdgeTpuInterpreter(*model, edgetpu_context.get()));
  }
  std::cout << 4 << std::endl;//TODO
  // Read the resized image file.
  int width, height, channels;
  const std::vector<uint8_t> input = read_bmp(resized_image_path, &width, &height, &channels);

  std::cout << 5 << std::endl;//TODO
  const auto required_shape = GetInputShape(*interpreter, 0);
  if (height != required_shape[0] || width != required_shape[1] || channels != required_shape[2]) {
    std::cerr << "Input size mismatches: "
      << "width: " << width << " vs " << required_shape[0] << ", height: " << height
      << " vs " << required_shape[1] << ", channels: " << channels << " vs "
      << required_shape[2] << std::endl;
    std::abort();
  }
  std::cout << 6 << std::endl;//TODO
  // Print inference result.
  const auto result = RunInference(input, interpreter.get());
  std::cout << 7 << std::endl;//TODO
  // Get Label
  const auto label = ParseLabel(label_path);
  std::cout << "8 " << label.size() << std::endl;//TODO
  size_t idx = 0;
  std::for_each(result.cbegin(), result.cend(), [&](const float& score) {
    if (score != 0) {
      std::cout << "------\n";
      std::cout << "Class: " << label.at(idx) << "\nScore: " << score << "\n";
    }
    idx++;
    });
  std::cout << 9 << std::endl;//TODO
  return 0;*/
}

