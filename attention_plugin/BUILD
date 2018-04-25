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

py_library(
    name = "attention_plugin",
    srcs = ["attention_plugin.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@org_pocoo_werkzeug",
        "@org_pythonhosted_six",
        "@org_tensorflow_tensorboard//tensorboard/backend:http_util",
        "@org_tensorflow_tensorboard//tensorboard/backend/event_processing:event_accumulator",
        "@org_tensorflow_tensorboard//tensorboard/plugins:base_plugin",
    ],
)

tf_web_library(
    name = "attention_dashboard",
    srcs = [
        "attention-card.html",
        "attention-dashboard.html",
    ],
    path = "/attention-plugin",
    deps = [
        "@org_polymer_iron_icon",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_input",
        "@org_tensorflow_tensorboard//tensorboard/components/tf_backend",
        "@org_tensorflow_tensorboard//tensorboard/components/tf_card_heading",
        "@org_tensorflow_tensorboard//tensorboard/components/tf_categorization_utils",
        "@org_tensorflow_tensorboard//tensorboard/components/tf_color_scale",
        "@org_tensorflow_tensorboard//tensorboard/components/tf_dashboard_common",
        "@org_tensorflow_tensorboard//tensorboard/components/tf_imports:lodash",
        "@org_tensorflow_tensorboard//tensorboard/components/tf_imports:polymer",
        "@org_tensorflow_tensorboard//tensorboard/components/tf_paginated_view",
        "@org_tensorflow_tensorboard//tensorboard/components/tf_runs_selector",
        "@org_tensorflow_tensorboard//tensorboard/components/tf_tensorboard:registry",
    ]
)