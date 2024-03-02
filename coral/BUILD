# Copyright 2019-2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//coral:model_benchmark_cases.bzl", "LSTM_MODEL_BENCHMARK_CASES", "MODEL_BENCHMARK_CASES")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "error_reporter",
    srcs = [
        "error_reporter.cc",
    ],
    hdrs = [
        "error_reporter.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/lite:stateful_error_reporter",
    ],
)

cc_library(
    name = "tflite_utils",
    srcs = [
        "tflite_utils.cc",
    ],
    hdrs = [
        "tflite_utils.h",
    ],
    deps = [
        "//coral/pose_estimation:posenet_decoder_op",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
        "@glog",
        "@libedgetpu//tflite/public:edgetpu",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:stateful_error_reporter",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
	"@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_binary(
    name = "coral",
    srcs = ["coraldevice.h", "coral.cpp"],
    linkshared=True,
    deps = [
        "//coral:tflite_utils",
        "//coral/classification:adapter",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@libedgetpu//tflite/public:oss_edgetpu_direct_all",  # buildcleaner: keep
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/c:common",
    ],
)

cc_binary(
    name = "main",
    srcs = ["coraldevice.h", "main.cpp"],
    deps = [
        "//coral:tflite_utils",
        "//coral/classification:adapter",
        "//coral/learn/backprop:softmax_regression_model",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@eigen_archive//:eigen3",
        "@flatbuffers",
        "@libedgetpu//tflite/public:oss_edgetpu_direct_all",  # buildcleaner: keep
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

