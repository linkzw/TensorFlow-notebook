#coding:utf-8
#导入相应模块
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
seed=2
 
 
def generateds():
	#基于seed生成随机数
	rdm = np.random.RandomState(seed)
	#随机数返回300行2列的矩阵
	X = rdm.randn(300,2)
	#从X中取出一行,判断x0 x1的平方和是否小于2,是则y赋值1,否则赋值0
	Y_ = [int (x1*x1 + x0*x0 <2) for (x0,x1) in X]	#这样获得的是一个list
	Y_c = [['red'if y  else 'blue'] for y in Y_]
	#调整Y_和Y_c的维度,第一个参数-1表示行数未知,第二个参数表示列数
	X = np.vstack(X).reshape(-1,2)
	Y_ = np.vstack(Y_).reshape(-1,1)
	
	return X, Y_, Y_c
	


