
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import dataset_factory
import model_deploy
import nets_factory
import preprocessing_factory

slim = tf.contrib.slim
framework = tf.contrib.framework
import tensorflow.contrib.slim.nets

network_fn = nets_factory.get_network_fn(
        'inception_v3',
        num_classes=5,
        weight_decay=0.00004,
        is_training=True)

model_variables = slim.get_model_variables()
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, './flowers/pretrained/inception_v3.ckpt')
    vars = sess.run(model_variables)
import tensorflow as tf
# from tensorflow.python import pywrap_tensorflow
#
# model_dir = "/Users/chentao/Desktop/transfer/slim/flowers/train_dir/inception_v3.ckpt"
#
# ckpt = tf.train.get_checkpoint_state(model_dir)
# ckpt_path = ckpt.model_checkpoint_path
#
# reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
# param_dict = reader.get_variable_to_shape_map()
#
# for key, val in param_dict.items():
#     try:
#         print
#         key, val
#     except:
#         pass
