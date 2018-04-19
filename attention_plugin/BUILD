package(default_visibility = ["//visibility:public"])

load("@org_tensorflow_tensorboard//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "attention_demo",
    srcs = ["attention_demo.py"],
    srcs_version = "PY2AND3",
    deps = [":attention_summary"],
)

py_library(
    name = "attention_summary",
    srcs = ["attention_summary.py", "attention_metadata.py"],
    srcs_version = "PY2AND3",
)