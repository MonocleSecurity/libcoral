// coraldevice.h
//

///// Includes /////

#include <cstddef>
#include <cstdint>

///// Defines /////

#ifdef _WIN32
#ifdef LIB_CORAL_DEVICE_EXPORT
#define LIB_CORAL_MODULE_API __declspec(dllexport)
#else
#define LIB_CORAL_MODULE_API
#endif
#else
#define LIB_CORAL_MODULE_API
#endif

///// Enumerations /////

enum class LIB_CORAL_DEVICE_TYPE : int
{
  USB,
  PCI
};

///// Declarations /////

struct LIB_CORAL_CONTEXT;
struct LIB_CORAL_DEVICE_CONTAINER;

///// Typedefs /////

typedef LIB_CORAL_CONTEXT*(*LIB_CORAL_INIT)(const char*);
typedef void(*LIB_CORAL_DESTROY)(LIB_CORAL_CONTEXT*);
typedef size_t(*LIB_CORAL_GET_NUM_DEVICES)(LIB_CORAL_CONTEXT*);
typedef LIB_CORAL_DEVICE_TYPE(*LIB_CORAL_GET_DEVICE_TYPE)(LIB_CORAL_CONTEXT*, const size_t);
typedef const char* (*LIB_CORAL_GET_DEVICE_PATH)(LIB_CORAL_CONTEXT*, const size_t);
typedef LIB_CORAL_DEVICE_CONTAINER*(*LIB_CORAL_OPEN_DEVICE)(LIB_CORAL_CONTEXT*, const size_t);
typedef void(*LIB_CORAL_CLOSE_DEVICE)(LIB_CORAL_DEVICE_CONTAINER*);
typedef const int* const(*LIB_CORAL_GET_INPUT_SHAPE)(LIB_CORAL_DEVICE_CONTAINER*); // returns an array of 3 integers
typedef const int* const(*LIB_CORAL_GET_OUTPUT_SHAPE)(LIB_CORAL_DEVICE_CONTAINER*); // returns an array of 3 integers
typedef int(*LIB_CORAL_RUN)(LIB_CORAL_DEVICE_CONTAINER*, const uint8_t* const, const size_t);
typedef const float* const(*LIB_CORAL_GET_RESULTS)(LIB_CORAL_DEVICE_CONTAINER*);
typedef size_t(*LIB_CORAL_GET_RESULTS_SIZE)(LIB_CORAL_DEVICE_CONTAINER*);

///// Prototypes /////

extern "C"
{
LIB_CORAL_MODULE_API LIB_CORAL_CONTEXT* LibCoralInit(const char* model);
LIB_CORAL_MODULE_API void LibCoralDestroy(LIB_CORAL_CONTEXT* context);

LIB_CORAL_MODULE_API size_t LibCoralGetNumDevices(LIB_CORAL_CONTEXT* context);
LIB_CORAL_MODULE_API LIB_CORAL_DEVICE_TYPE LibCoralGetDeviceType(LIB_CORAL_CONTEXT* context, const size_t index);
LIB_CORAL_MODULE_API const char* LibCoralGetDevicePath(LIB_CORAL_CONTEXT* context, const size_t index);

LIB_CORAL_MODULE_API LIB_CORAL_DEVICE_CONTAINER* LibCoralOpenDevice(LIB_CORAL_CONTEXT* context, const size_t index);
LIB_CORAL_MODULE_API void LibCoralCloseDevice(LIB_CORAL_DEVICE_CONTAINER* device);
LIB_CORAL_MODULE_API const int* const LibCoralGetInputShape(LIB_CORAL_DEVICE_CONTAINER* device);
LIB_CORAL_MODULE_API const int* const LibCoralGetOutputShape(LIB_CORAL_DEVICE_CONTAINER* device);
LIB_CORAL_MODULE_API int LibCoralRun(LIB_CORAL_DEVICE_CONTAINER* device, const uint8_t* const data, const size_t size);
LIB_CORAL_MODULE_API const float* const LibCoralGetResults(LIB_CORAL_DEVICE_CONTAINER* device);
LIB_CORAL_MODULE_API size_t LibCoralGetResultsSize(LIB_CORAL_DEVICE_CONTAINER* device);
}
