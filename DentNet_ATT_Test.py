import os
import tensorflow as tf
from tensorflow.contrib.layers import flatten, batch_norm# FC
from tensorflow.contrib.framework import arg_scope # BN

def Batch_Normalization(x, training, scope="batch_norm"):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        if training:
            return  batch_norm(inputs=x, is_training=training, reuse=None)
        else:
            return batch_norm(inputs=x, is_training=training, reuse=tf.AUTO_REUSE)

def maxPoolLayer(x,kHeight,kWidth,strideX,strideY,name,padding="VALID"):
    with tf.variable_scope(name):
        return tf.nn.max_pool(x,ksize=[1,kHeight,kWidth,1],strides=[1,strideX,strideY,1],padding=padding,name=name)

def convLayer(x,kHeight,kWidth,strideX,strideY,featureNum,name,pname="n",padding='SAME'):
    with tf.variable_scope(name):
        channel = int(x.get_shape()[-1])  # Get the number of channels of the input
        w = tf.get_variable(name+"_w",shape=[kHeight,kWidth,channel,featureNum],initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name+"_b",shape=[featureNum],initializer=tf.constant_initializer(0.0))
        featureMap = tf.nn.conv2d(x,w,strides=[1,strideX,strideY,1],padding=padding)
        out = tf.nn.bias_add(featureMap,b)
        if len(pname)>1:
            out = prelu(out,name)

        return out

def conv1x1Layer(x,kHeight,kWidth,strideX,strideY,featureNum,name,padding='VALID'):
    with tf.variable_scope(name):
        channel = int(x.get_shape()[-1])  # Get the number of channels of the input
        w = tf.get_variable(name+"_w",shape=[kHeight,kWidth,channel,featureNum],initializer=tf.contrib.layers.xavier_initializer())
        featureMap = tf.nn.conv2d(x,w,strides=[1,strideX,strideY,1],padding=padding)

        return featureMap

def concatLayer(inputs_x, inputs_y, inputs_z, is_train, name):
    with tf.variable_scope(name):
        out_shape = inputs_x.get_shape().as_list()
        inputs_x = tf.contrib.layers.batch_norm(inputs_x, scale=True, is_training=is_train, updates_collections=None)
        inputs_y = tf.contrib.layers.batch_norm(inputs_y, scale=True, is_training=is_train, updates_collections=None)
        inputs_z = tf.contrib.layers.batch_norm(inputs_z, scale=True, is_training=is_train, updates_collections=None)

        p = tf.get_variable("p", [3, 1, out_shape[2], out_shape[3]], initializer=tf.ones_initializer(), trainable=True)
        p = tf.nn.softmax(p, 0)
        output_layer = p[0:1, :, :, :] * inputs_x + p[1:2, :, :, :] * inputs_y + p[2:3, :, :, :] * inputs_z
        output_layer = prelu(output_layer, name + "adaprelu")

        return output_layer

def fcLayer(x,inputD,outputD,name,pname="f"):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable(name+"_w",shape=[inputD,outputD],dtype="float",initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name+"_b",[outputD],dtype="float",initializer=tf.constant_initializer(0.0))
        out = tf.nn.xw_plus_b(x,w,b,name=scope.name)
        if len(pname)>1:
            out = prelu(out,name)

        return out

#prelu activation function Note that you need to change is_train to False when testing
def prelu(_x, scope=None, name ='Prelu',is_train=False):
    #"""parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _x = Batch_Normalization(_x,is_train,name +"BN") 
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.25))
        
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)
def dropout(x,keep_prob,name):
    with tf.variable_scope(name):
        return tf.nn.dropout(x,keep_prob,name=name)


def attentional_transition(input_layer,name):
    channels = input_layer.get_shape().as_list()[-1]
    bottom_fc = tf.nn.avg_pool(input_layer, [1, 16, 16, 1], [1, 16, 16, 1], 'VALID') # 16 Depends on the size of the feature map

    assert bottom_fc.get_shape().as_list()[-1] == channels #none,1,1,c
    bottom_fc=tf.reshape(bottom_fc, [-1, channels])   ## none, C

    Wfc = tf.get_variable(name=name+'_W1',shape=[channels,channels/2],initializer=tf.contrib.layers.xavier_initializer())
    bfc = tf.get_variable(name=name+'_b1',initializer=tf.constant(0.0,shape=[channels/2]))
    mid_fc = prelu(tf.matmul(bottom_fc, Wfc)+bfc, name+"_fcprelu")

    Wfc=tf.get_variable(name=name+'_W2', shape=[channels/2, channels], initializer=tf.contrib.layers.xavier_initializer())
    bfc=tf.get_variable(name=name+'_b2', initializer=tf.constant(0.0, shape=[channels]))
    top_fc=tf.nn.sigmoid(tf.matmul(mid_fc, Wfc)+bfc)   ## none, C
    top_fc = tf.reshape(top_fc, [-1, 1, 1, channels])
    output_layer = tf.multiply(input_layer, top_fc)

    return output_layer

def transition(input_layer,if_a,is_train,name):
    with tf.variable_scope(name):
        channels = input_layer.get_shape().as_list()[-1]
        output_layer = tf.contrib.layers.batch_norm(input_layer,scale=True,is_training=is_train,updates_collections=None)
        output_layer = prelu(output_layer,name=name+'Prelu',is_train=is_train)
        output_layer = convLayer(output_layer,1,1,1,1,channels,name=name,padding='SAME')
        if if_a:
            output_layer = attentional_transition(output_layer,name=name+'-ATT')

        return output_layer

class DentNet_ATT:
    def __init__(self, x, classNum, is_train=True, modelPath=None):
        self.X = x
        self.CLASSNUM = classNum
        self.is_train = is_train
        self.if_a = True
        #self.keep_prob = 0.7
        self.MODLEPATH = modelPath
        self.build()

    def build(self):
        self.conv1_1 = convLayer(self.X, 3, 3, 1, 1, 64, "conv1_1", "prelu1_1")  # convolution kernel height, width, stride, stride, output channel
        self.conv1_2 = convLayer(self.conv1_1, 3, 3, 1, 1, 64, "conv1_2", "prelu1_2")

        self.pool1_1 = maxPoolLayer(self.conv1_2, 2, 2, 2, 2, "pool1")

        self.conv2_1 = convLayer(self.pool1_1, 3, 3, 1, 1, 128, "conv2_1", "prelu2_1")
        self.conv2_2 = convLayer(self.conv2_1, 3, 3, 1, 1, 128,"conv2_2", "prelu2_2")
        self.conv2_3 = convLayer(self.conv2_2, 3, 3, 1, 1, 128,"conv2_3", "prelu2_3")
        self.pool2_1 = maxPoolLayer(self.conv2_3, 2, 2, 2, 2, "pool2")

        self.conv3_1 = convLayer(self.pool2_1, 3, 3, 1, 1, 128,"conv3_1", "prelu3_1")
        self.conv3_2 = convLayer(self.conv3_1, 3, 3, 1, 1, 128,"conv3_2", "prelu3_2")
        self.conv3_3 = convLayer(self.conv3_2, 3, 3, 1, 1, 128,"conv3_3", "prelu3_3")
        self.pool3_1 = maxPoolLayer(self.conv3_3, 2, 2, 2, 2, "pool3")

        self.conv1x1 = conv1x1Layer(self.pool3_1,1,1,1,1,256,"conv1x1")

        self.trainsition1_1 = transition(self.conv1x1,self.if_a, is_train=self.is_train, name='transition1_1')

        self.conv4_1 = convLayer(self.conv1x1,3,3,1,1,256,"conv4_1","prelu4_1")
        self.conv4_2 = convLayer(self.conv4_1,3,3,1,1,256,"conv4_2","prelu4_2")
        self.conv4_3 = convLayer(self.conv4_2,3,3,1,1,256,"conv4_3","prelu4_3")

        self.trainsition1_2 = transition(self.conv4_3,self.if_a, is_train=self.is_train, name='transition1_2')

        self.conv5_1 = convLayer(self.conv4_3, 3, 3, 1, 1, 256,"conv5_1", "prelu5_1")
        self.conv5_2 = convLayer(self.conv5_1, 3, 3, 1, 1, 256,"conv5_2", "prelu5_2")
        self.conv5_3 = convLayer(self.conv5_2, 3, 3, 1, 1, 256,"conv5_3", "prelu5_3")

        self.trainsition1_3 =transition(self.conv5_3,self.if_a, is_train=self.is_train, name='transition1_3')
        self.eltw_layer = concatLayer(self.trainsition1_1, self.trainsition1_2, self.trainsition1_3,is_train=self.is_train, name="eltw_layer")

        self.conv6_1 = convLayer(self.eltw_layer, 3, 3, 1, 1, 512, "conv6_1", "prelu6_1")
        self.conv6_2 =  convLayer(self.conv6_1, 3, 3, 1, 1, 512,"conv6_2", "prelu6_2")
        self.conv6_3 =  convLayer(self.conv6_2, 3, 3, 1, 1, 512,"conv6_3", "prelu6_3")

        self.fc_Input = flatten(self.conv6_3)

        self.fc1 = fcLayer(self.fc_Input, 16 * 16 * 512, 1024, "fc1", "fc1_prelu")
        self.feature = fcLayer(self.fc1, 1024, 512, "fc2", "fc2_prelu")

        self.fc3 = fcLayer(self.feature,512,self.CLASSNUM,"fc3")
        self.logits = tf.nn.softmax(self.fc3)
