"""Simple demo which greets several people."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf

import attention_summary

# Directory into which to write tensorboard data.
LOGDIR = '/tmp/attention_demo'


def run(logdir, run_name, input_strings, output_strings, attention_dists):
    """Greet several characters from a given cartoon."""

    tf.reset_default_graph()

    input_ = tf.placeholder(tf.string)
    output_ = tf.placeholder(tf.string)
    attention_ = tf.placeholder(tf.float32)

    input_op = attention_summary.input_op("attention", input_)
    output_op = attention_summary.output_op("attention", output_)
    attention_op = attention_summary.attention_op("attention", attention_)

    writer = tf.summary.FileWriter(os.path.join(logdir, run_name))

    sess = tf.Session()

    for i in range(len(input_strings)):
        input_string = input_strings[i]
        output_string = output_strings[i]
        attention_dist = attention_dists[i]

        input_summary, output_summary, attn_summary = sess.run([input_op, output_op, attention_op], feed_dict={input_:                     input_string, output_: output_string, attention_: attention_dist})

        writer.add_summary(input_summary)
        writer.add_summary(output_summary)
        writer.add_summary(attn_summary)

    writer.close()



def run_all(logdir, unused_verbose=False):
    """
    Run the simulation for every logdir.
    """
    run(logdir, "demo", ["A", "Text", "Sequence"], ["Okay", "Hello", "Hi"],
        [[1], [2], [1]])
 
def main(unused_argv):
    print('Saving output to %s.' % LOGDIR)
    run_all(LOGDIR, unused_verbose=True)
    print('Done. Output saved to %s.' % LOGDIR)


if __name__ == '__main__':
  tf.app.run()