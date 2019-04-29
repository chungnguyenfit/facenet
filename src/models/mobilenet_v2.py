from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow_hub as hub




'''
def mobilenetV2_model(images, keep_probability, phase_train=True, bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    from slim.nets.mobilenet import mobilenet_v2
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=phase_train)):
        logits, endpoints = mobilenet_v2.mobilenet(images)
    
    return logits
'''
        
'''
def mobilenetV2_model(images, keep_probability, phase_train=True, bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    #saver = tf.train.import_meta_graph('../model/backbone/mobilenet_v2_1.4_224.ckpt.meta')
    #saver.restore(sess,tf.train.latest_checkpoint('./'))
    module = hub.Module("https://tfhub.dev/google/imagenet/mobilenet_v2_100_160/feature_vector/2", trainable=True)
    print(module.get_signature_names())
    height, width = hub.get_expected_image_size(module)
    features = module(images)  # Features with shape [batch_size, num_features].
    return features
'''
def mobilenetV2_model(images, keep_probability, phase_train=True, bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    '''
    from . import mobiletnet_v2_utils
    logits, endpoints = mobiletnet_v2_utils.mobilenet_v2(images,
                     num_classes=1000,
                     dropout_keep_prob=keep_probability,
                     is_training=phase_train,
                     depth_multiplier=1.0,
                     prediction_fn=tf.contrib.layers.softmax,
                     spatial_squeeze=True,
                     scope='MobilenetV2')
    '''
    
    from . import mobiletnet_v2_utils2
    logits, endpoints = mobiletnet_v2_utils2.mobilenetv2(inputs=images, num_classes=1000, is_train=phase_train, reuse=reuse)
    
    '''
    mobilenet = tf.keras.applications.MobileNetV2(input_tensor=images, include_top=False, pooling='avg', alpha=1, weights = None)
    for layer in mobilenet.layers:
        layer.trainable = True  # trainable has to be false in order to freeze the layers

    logits = mobilenet.output
    '''
    return logits
    
def inference(images, keep_probability, phase_train=True, bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with tf.variable_scope('mobilenet_v2', [images], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=phase_train):

                net = mobilenetV2_model(images, keep_probability, phase_train, bottleneck_layer_size, weight_decay, reuse)
                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, 
                        scope='Bottleneck', reuse=False)
                
    return net, None

