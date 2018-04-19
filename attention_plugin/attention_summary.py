from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import attention_metadata

def input_op(name, data, display_name=None, description=None, collections=None):
    if (display_name == None):
        display_name = name

    summary_metadata = attention_metadata.create_summary_metadata(
        display_name=display_name, description=description)

    with tf.name_scope(name):
        with tf.control_dependencies([tf.assert_type(data, tf.string)]):
            return tf.summary.tensor_summary(name='attention_input_summary',
                                            tensor=data,
                                            collections=collections,
                                            summary_metadata=summary_metadata)

def output_op(name, data, display_name=None, description=None, collections=None):
    if (display_name == None):
        display_name = name

    summary_metadata = attention_metadata.create_summary_metadata(
        display_name=display_name, description=description)

    with tf.name_scope(name):
        with tf.control_dependencies([tf.assert_type(data, tf.string)]):
            return tf.summary.tensor_summary(name='attention_output_summary',
                                            tensor=data,
                                            collections=collections,
                                            summary_metadata=summary_metadata)

def attention_op(name, data, display_name=None, description=None, collections=None):
    if (display_name == None):
        display_name = name

    summary_metadata = attention_metadata.create_summary_metadata(
        display_name=display_name, description=description)

    with tf.name_scope(name):
        with tf.control_dependencies([tf.assert_type(data, tf.float32)]):
            return tf.summary.tensor_summary(name='attention_dist_summary',
                                            tensor=data,
                                            collections=collections,
                                            summary_metadata=summary_metadata)


def input_pb(name, data, display_name=None, description=None):
    try:
        tensor = tf.make_tensor_proto(data, dtype=tf.string)
    except TypeError as e:
        raise ValueError(e)

    if display_name is None:
        display_name = name

    summary_metadata = attention_metadata.create_summary_metadata(
        display_name=display_name, description=description)

    summary = tf.Summary()
    summary.value.add(tag='%s/attention_input_summary' % name,
                      metadata=summary_metadata,
                      tensor=tensor)

def output_pb(name, data, display_name=None, description=None):
    try:
        tensor = tf.make_tensor_proto(data, dtype=tf.string)
    except TypeError as e:
        raise ValueError(e)

    if display_name is None:
        display_name = name

    summary_metadata = attention_metadata.create_summary_metadata(
        display_name=display_name, description=description)

    summary = tf.Summary()
    summary.value.add(tag='%s/attention_output_summary' % name,
                      metadata=summary_metadata,
                      tensor=tensor)

def attention_pb(name, data, display_name=None, description=None):
    try:
        tensor = tf.make_tensor_proto(data, dtype=np.float32)
    except TypeError as e:
        raise ValueError(e)

    if display_name is None:
        display_name = name

    summary_metadata = attention_metadata.create_summary_metadata(
        display_name=display_name, description=description)

    summary = tf.Summary()
    summary.value.add(tag='%s/attention_dist_summary' % name,
                      metadata=summary_metadata,
                      tensor=tensor)