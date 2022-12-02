import tensorflow as tf
import cv2
import tensorflow_io as tfio
import pathlib
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label, regionprops, regionprops_table
import itertools
Image.MAX_IMAGE_PIXELS = 570000000
ksize_rows = 32
ksize_cols = 32
strides_rows = 32
strides_cols = 32
train_split=0.8
val_split=0.1
test_split=0.1
import numpy as np

def preprocess_image(paths):
    # # image= tf.io.read_file(image_path)
    image_path=paths[0]
    mask_path=paths[1]
    # image= cv2.imread(image_path.astype('str'))
    o=Image.open(image_path)
    o.convert('RGB')
    m= Image.open(mask_path)
    m.convert('RGB')
    # imarray = tfio.experimental.image.decode_tiff(image)
    imarrayo = np.array(o)
    imarrym= np.array(m)
    orgBinary = imarrayo > 0
    orgBinary = label(orgBinary)
    props = regionprops(orgBinary.astype(np.uint8))[0]
    imarrayo = imarrayo[props.bbox[0]:props.bbox[2] , props.bbox[1]:props.bbox[3]]
    imarraym = imarrym[props.bbox[0]:props.bbox[2] , props.bbox[1]:props.bbox[3]]
    # print(imarrayo.shape)

    # The size of sliding window
    ksizes = [1, ksize_rows, ksize_cols, 1]

    # How far the centers of 2 consecutive patches are in the image
    strides = [1, strides_rows, strides_cols, 1]
    rates = [1, 1, 1, 1]  # sample pixel consecutively

    # padding algorithm to used
    padding = 'VALID'  # or 'SAME'

    if len(imarrayo.shape) == 2:
        imarrayo = np.tile(imarrayo[:, :, np.newaxis], (1, 1, 3))
        imarraym = np.tile(imarraym[:, :, np.newaxis], (1, 1, 3))

    # impng=tf.image.decode_png(imarray, channels=1)
    imarrayo =tf.expand_dims(imarrayo, axis=0)
    # imarrayo = tf.expand_dims(imarrayo, axis=-1)

    imarraym = tf.expand_dims(imarraym, axis=0)
    # imarraym = tf.expand_dims(imarraym, axis=-1)


    image_patches = tf.image.extract_patches(imarrayo, ksizes, strides, rates, padding)
    mask_patches= tf.image.extract_patches(imarraym, ksizes, strides, rates, padding)
    # print(image_patches.shape, mask_patches.shape)  # => TensorShape([125, 5, 5, 262144]) . Why we have 5 pictures in a row?
    # #
    # patch1 = image_patches[0, 0, 0,]  # Get the 1st patch
    # patch2= image_patches [0,0,1,]
    # patch3= image_patches[0,0,2,]
    # patch8= image_patches[0,0,3,]
    # patch4= image_patches[0,0,5]
    # patch5= image_patches[0,0,6,]
    # patch6=image_patches[0,0,7,]
    # patch7= image_patches[0,0,8]
    # patch1 = tf.reshape(patch1, [32, 32, 1])  # Reshape to the correct shape
    # patch1 = tf.squeeze(patch1)  # Remove the depth channel
    #
    # patch2 = tf.reshape(patch2, [32, 32, 1])  # Reshape to the correct shape
    # patch2 = tf.squeeze(patch2)  # Remove the depth channel
    #
    # patch3 = tf.reshape(patch3, [32, 32, 1])  # Reshape to the correct shape
    # patch3 = tf.squeeze(patch3)  # Remove the depth channel
    #
    # patch4 = tf.reshape(patch4, [32, 32, 1])  # Reshape to the correct shape
    # patch4 = tf.squeeze(patch4)  # Remove the depth channel
    #
    # patch5 = tf.reshape(patch5, [32, 32, 1])  # Reshape to the correct shape
    # patch5 = tf.squeeze(patch5)  # Remove the depth channel
    #
    # patch6 = tf.reshape(patch6, [32, 32, 1])  # Reshape to the correct shape
    # patch6 = tf.squeeze(patch6)  # Remove the depth channel
    #
    # patch7 = tf.reshape(patch7, [32, 32, 1])  # Reshape to the correct shape
    # patch7 = tf.squeeze(patch7)  # Remove the depth channel
    #
    # patch8 = tf.reshape(patch8, [32, 32, 1])  # Reshape to the correct shape
    # patch8 = tf.squeeze(patch8)  # Remove the depth channel
    # plt.imsave('patch3.tiff', patch1)
    # plt.imsave('patch4.tiff', patch2)
    # plt.imsave('patch5.tiff', patch3)
    # plt.imsave('patch6.tiff', patch4)
    # plt.imsave('patch7.tiff', patch5)
    # plt.imsave('patch8.tiff', patch6)
    # plt.imsave('patch9.tiff', patch7)
    # plt.imsave('patch10.tiff', patch8)
    #






    return  tf.reshape(image_patches, (-1,32,32,3)), tf.reshape(mask_patches, (-1,32,32,3))



def get_data(dataset_path):
    data_root = pathlib.Path(dataset_path)

    all_image_paths = data_root.glob('images/*')
    all_image_paths = [str(path) for path in all_image_paths]

    all_mask_paths = data_root.glob('annotations/*')
    all_mask_paths = [str(path) for path in all_mask_paths]
    image_ds = list(map(preprocess_image, zip(all_image_paths, all_mask_paths)))
    images = image_ds[0]
    image_shape = images[0].shape
    ds_size = image_shape.dims[0].value
    image_pairs = [x[0] for x in image_ds]
    mask_pairs = [x[1] for x in image_ds]
    image_pairs = tf.data.Dataset.from_tensor_slices(image_pairs[0])
    mask_pairs = tf.data.Dataset.from_tensor_slices(mask_pairs[0])
    # ds = tf.data.Dataset.zip((image_pairs, mask_pairs))

    # train_size = int(train_split * ds_size)
    # val_size = int(val_split * ds_size)
    # train_ds_images=image_pairs[0:train_size]
    # # train_ds_images = tf.data.Dataset.from_tensor_slices(image_pairs.take(train_size))
    # val_ds_images = tf.data.Dataset.from_tensor_slices(image_pairs.skip(train_size).take(val_size))
    # test_ds_images = image_pairs.skip(train_size).skip(val_size)
    # train_ds_masks = mask_pairs.take(train_size)
    # val_ds_masks = mask_pairs.skip(train_size).take(val_size)
    # test_ds_masks = mask_pairs.skip(train_size).skip(val_size)

    return (image_pairs, mask_pairs)

train_ds_images, train_ds_masks = get_data(r'C:\Users\P301644\Documents\simclr\mitochondria') #later look at k-fold
# for image, mask in image_mask_ds.take(1):
#     sample_image, sample_mask = image, mask
#
# display_sample([sample_image.numpy(), sample_mask.numpy()])

