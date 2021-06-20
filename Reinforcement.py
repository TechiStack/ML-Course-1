import os
import numpy as np
import tensorflow as tf


class DeepQNetwork(object):
    def __init__(self,lr,n_actions,name,fcl_dims = 256 , input_dims = (210,160,4),chktpt_dir='C:\Users\zedge\Documents'):
        self.lr  = lr
        self.n_actions = n_actions
        self.name = name
        self.fcl_dims = fcl_dims
        self.input_dims = input_dims
        self.sess = tf.Session()
        self.build_network()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver
        self.checkpoint_file  = os.path.join(chktpt_dir,'deepqnet.ckpt')
        self.params = tf.get_collections(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope = self.name)

    def build_net(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32 , shape = [None , *self.input_dims] , 
                                        name = 'input')
            self.actions = tf.placeholder(tf.float32 , shape = [None, self.n_actions,] ,  
                                        name = 'n_actions')
              
            self.q_targets = tf.placeholder(tf.float32,shape = [None,self.n_actions])

            conv1 = tf.layers.conv2d(inputs = self.input, 
                                    filters = 32 ,kernelsize = (8,8) , 
                                    stride = 4 , name = 'conv1',
                                    kernel_initializer = tf.variances_scaling_initializer(scale=2))
            conv1_activated = tf.nn.relu(conv1)                        
            conv2  = tf.layers.conv2d(inputs = conv1_activated . filters = 64 , kernel_size = (4,4)
                                    ,stride = 2 , name = 'conv2')
            conv2_activated = tf.nn.relu(conv2)
            
            conv3  = tf.layers.conv2d(inputs = conv2_activated , filters =128 , kernel_size = (3,3),stride = 1 , name = 'conv3',
                                    kernel_initializer = tf.variances_scaling_initializer(scale=2))
            conv3_activated = tf.nn.relu(conv3)








