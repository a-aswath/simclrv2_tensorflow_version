import re
import numpy as np
import pickle
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow_hub as hub
from cluster import umap_dimension_reduction, cluster_KMeans
from mitochondria.get_image_data import get_data
import tensorflow_datasets as tfds
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib
import matplotlib.pyplot as plt


def count_params(checkpoint, excluding_vars=[], verbose=True):
  vdict = checkpoint.get_variable_to_shape_map()
  cnt = 0
  for name, shape in vdict.items():
    skip = False
    for evar in excluding_vars:
      if re.search(evar, name):
        skip = True
    if skip:
      continue
    if verbose:
      print(name, shape)
    cnt += np.prod(shape)
  cnt = cnt / 1e6
  print("Total number of parameters: {:.2f}M".format(cnt))
  return cnt

# checkpoint_path = r'C:\Users\P301644\Documents\simclr\pretrained_model\ckpt\simclrv2_pretrained_r50_1x_sk0_model.ckpt-250228'
# checkpoint = tf.train.load_checkpoint(checkpoint_path)
# _ = count_params(checkpoint, excluding_vars=['global_step', "Momentum", 'ema', 'memory', 'head'], verbose=False)


#get data-- tf.data.Dataset.from_tensor_slices
batch_size=1
train_ds_images, train_ds_masks = get_data(r'C:\Users\P301644\Documents\simclr\mitochondria')
num_train_examples = train_ds_images._tensors[0].shape.dims[0]
x = train_ds_images.batch(batch_size)
y= tf.data.make_one_shot_iterator(x).get_next()
# Load hub module for inference

hub_path = r'C:\Users\P301644\Documents\simclrv2_tensorflow_version\pre_trained_model\r50_1x_sk0\hub'
module = hub.Module(hub_path, trainable=True)
key = module(inputs=tf.cast(y, tf.float32), signature="default", as_dict=True)
logits_t = key['block_group4'][:, :, :, :]
iter_tensor_patches = tf.data.make_one_shot_iterator(x)
# x = tf.data.get_next()

pred_list = []
df= pd.DataFrame()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:



    sess.run(tf.global_variables_initializer())
    for idx, batch in enumerate(tqdm(range(num_train_examples // batch_size))):
      input = tf.cast(iter_tensor_patches.get_next(), tf.float32)
      image, labels, logits = sess.run((input, [], logits_t))
      # pred = logits.argmax(-1)
      df[idx]= np.column_stack(np.squeeze(logits))
df.to_feather('embedding%d.feather', idx)
#
#
# import umap
# def umap_d(nr_dimensions, data):
#     reduced_data = umap.UMAP(n_components=nr_dimensions, metric='euclidean', init='random').fit(data)
#     return reduced_data
# trans = umap_d(10,pred_list)
#
# plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 5, c=[], cmap='Spectral')
# plt.title('Embedding of the training set by UMAP', fontsize=24);
# reduced_embedding = umap_dimension_reduction(4, pred_list)
# clutered_data= cluster_KMeans()

# print(module.get_signature_names())
# print(module.get_input_info_dict())
# print(module.get_output_info_dict())
# print(tf.all_variables()) # Find out the names of all variables present in graph

# features = module["default"]

# with tf.variable_scope('CustomLayer'):
#     weight = tf.get_variable('weights', initializer=tf.truncated_normal((2048, 2)))
#     bias = tf.get_variable('bias', initializer=tf.ones((2)))
#     logits = tf.nn.xw_plus_b(features, weight, bias)
#
#
#
# https://medium.com/ymedialabs-innovation/how-to-use-tensorflow-hub-with-code-examples-9100edec29af
# # After finding the names of variables or scope, gather the variables you wish to fine-tune
# var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CustomLayer')
# var_list2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='module/resnet_v2_50/block4')
# var_list.extend(var_list2)


#
# logits_t = key['logits_sup'][:, :]
# softmax = tf.nn.softmax(logits_t)
# top_predictions = tf.nn.top_k(softmax, top_k, name='top_predictions')
# key # The accessible tensor in the return dictionary