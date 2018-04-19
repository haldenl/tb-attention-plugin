from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Internal information about the attention plugin."""
import tensorflow as tf
from tensorboard.plugins.text import plugin_data_pb2


PLUGIN_NAME = 'attention'

def create_summary_metadata(display_name, description):
  """Create a `tf.SummaryMetadata` proto for attention plugin data.
  Returns:
    A `tf.SummaryMetadata` protobuf object.
  """
  metadata = tf.SummaryMetadata(
      display_name=display_name,
      summary_description=description,
      plugin_data=tf.SummaryMetadata.PluginData(
          plugin_name=PLUGIN_NAME))

  return metadata