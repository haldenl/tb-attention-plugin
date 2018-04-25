import tensorflow as tf
import numpy as np
import six
from werkzeug import wrappers

from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin

class AttentionPlugin(base_plugin.TBPlugin):
    """A plugin that serves attention summaries recorded during model runs"""

    plugin_name = 'attention'

    def __init__(self, context):
        """Instantiates an AttentionPlugin"""
        self._multiplexer = context.multiplexer

    @wrappers.Request.application
    def tags_route(self, request):
        """
        A route (HTTP handler) that returns a response with tags.

        Returns:
            A response that contains a JSON object. The keys of the
            object are all the runs. Each run is mapped to a
            (potentially empty) list of all tags that are
            relevant to this plugin
        """
        all_runs = self._multiplexer.PluginRunToTagToContent(
            AttentionPlugin.plugin_name)
        
        response = {
            run: tagToContent.keys()
                    for (run, tagToContent) in all_runs.items()
        }

        return http_util.Respond(request, response, 'application/json')

    def get_plugin_apps(self):
        """
        Gets all routes offered by the plugin.

        This method is called by TensorBoard when retrieving all the
        routes offered by the plugin.

        Returns:
            A dictionary mapping URL path to route that handles it.
        """
        return {
            '/tags': self.tags_route,
            '/attention': self.attention_route,
        }

    def is_active(self):
        """
        Determines whether this plugin is active.

        This plugin is only active if TensorBoard sampled any summaries
        relevant to the greeter plugin.

        Returns:
            Whether this plugin is active.
        """
        all_runs = self._multiplexer.PluginRunToTagToContent(
            AttentionPlugin.plugin_name)

        return bool(self._multiplexer and any(six.itervalues(all_runs)))

    @wrappers.Request.application
    def attention_route(self, request):
        """
        A route that returns the attention summaries associated
        with a tag.

        Returns:
            A JSON list of summaries associated with run and tag
            combination. The summaries take the form of
            {
                input: []
                output: []
                attn_dist: []
            }
        """
        run = request.args.get('run')
        tag = request.args.get('tag')

        tensor_events = self._multiplexer.Tensors(run, tag)
        response = [self._process_string_tensor_event(ev) for ev in tensor_events]

        return http_util.Respond(request, response, 'application/json')
        
    def _process_string_tensor_event(self, event):
        """Convert a TensorEvent into a JSON-compatible response."""
        string_arr = tf.make_ndarray(event.tensor_proto)
        text = string_arr.astype(np.dtype(str)).tostring()
        return {
            'wall_time': event.wall_time,
            'step': event.step,
            'text': text,
        }