#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 15:31:47 2018

@author: Mir, A.
"""

# In this module, the installation of TensorFlow will be tested.

from tensorflow.python.client import device_lib
import tensorflow as tf


def get_gpu_info():
    
    """
    Returns informaion on available GPUs on system
    """
    
    local_gpus = device_lib.list_local_devices()
    
    # GPU device information
    gpu_info = [dev.physical_device_desc for dev in local_gpus if dev.device_type == 'GPU']
    
    return ''.join([str_info.strip() + '\n' for str_info in gpu_info[0].split(',')])



print(get_gpu_info())

# GPU options to limit memory allocated
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

hello = tf.constant('Hello, TensorFlow!!')

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as tf_ses:
    
    tf_ses.run(hello)
