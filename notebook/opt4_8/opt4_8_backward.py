#coding:utf-8
#定义反向传播过程

#导入相应模块
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import opt4_8_generateds
import opt4_8_forward
#超参数设置 
STEPS = 40000
BASH_SIZE = 300
LREANING_RATE_BASH = 0.001
LREANING_RATE_DECAY = 0.999
GULARIZER = 0.01


#定义反向传播过程
def backward():
	#占位
	x = tf.placeholder(tf.float32, shape = (None, 2))
	y_ = tf.placeholder(tf.float32, shape = (None, 1))
	#参数获取
	X, Y_, Y_c=opt4_8_generateds.generateds()
	y = opt4_8_forward.forward(x, GULARIZER)
	#计数器初始化
	golbal_step = tf.Variable(0, trainable = False)
	#生成指数衰减学习率
	lreaning_rate = tf.train.exponential_decay(LREANING_RATE_BASH, golbal_step, BASH_SIZE, LREANING_RATE_DECAY, staircase = True)
	
	#定义损失函数
	loss_mse = tf.reduce_mean(tf.square(y_-y))
	loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))
	#使用梯度下降法
	train_step = tf.train.AdamOptimizer(lreaning_rate).minimize(loss_total)

	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		for i in range(STEPS):
			start = (i*BASH_SIZE) % 300
			end = start + BASH_SIZE
			sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
			#if i % 2000 == 0
		xx, yy = np.mgrid[-3:3:.01, -3:3:0.1]
		grid = np.c_[xx.ravel(), yy.ravel()]
		probs = sess.run(y, feed_dict={x:grid})
		probs = probs.reshape(xx.shape)

	plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
	plt.contour(xx, yy, probs, levels = [.5])
	plt.show()

if __name__=='__main__':
	backward()

