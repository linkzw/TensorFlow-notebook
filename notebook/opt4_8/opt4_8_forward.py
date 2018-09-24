#coding:utf-8
#定义前向传播过程

#导入相关模块
import tensorflow as tf
import numpy as np

def get_weight(shape, regularizer):
	w = tf.Variable(tf.random_normal(shape), dtype = tf.float32)
	#引入l2正则化
	tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

def get_bias(shape):
	b = tf.Variable(tf.constant(0.01, shape = shape))
	return b

def forward(x, regularizer):
	#隐藏层
	w1 = get_weight([2,11], regularizer)
	b1 = get_bias([11])
	y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
	#输出层
	w2 = get_weight([11,1], regularizer)
	b2 = get_bias([1])
	y = tf.matmul(y1, w2) + b2		#输出层不过激活
	return y
