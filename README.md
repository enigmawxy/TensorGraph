# TensorGraph
## 纯Python对<a href="http://www.tensorflow.org">TensorFlow</a>功能的重新实现

TensorGraph是一个模仿TensorFlow API对小的机器学习API，使用纯Python实现，没有C. 代码实现考虑的是概念上的理解，而不是考虑功能实现的效率. 所以，只适用于学习的目的. 如果你想理解像TensorFlow这样的深度学习库的工作机制，这个项目是非常适合的. 

我在CSDN上写了博客 <a href="http://www.csdn.net/">CSDN</a> 来讲述如何开始这个项目，可以浏览。

## 使用方法
导入:

    import TensorGraph as tg

创建计算图:

    tg.Graph().as_default()

创建输入占位符:

    training_features = tg.placeholder()
    training_classes = tg.placeholder()

建一个神经网络结构:

	weightg = tg.Variable(np.random.randn(2, 2))
	biases = tg.Variable(np.random.randn(2))
	model = tg.softmax(tg.add(tg.matmul(X, W), b))

创建一个优化函数或者损失函数:

    loss = tg.negative(tg.reduce_sum(tg.reduce_sum(tg.multiply(training_classes, tg.log(model)), axis=1)))

选择使用优化器:

    optimizer = tg.train.GradientDescentOptimizer(learning_rate=0.01).minimize(J)

向输入占位符输入数据:

	feed_dict = {
		training_features: my_training_features,
		training_classes: my_training_classes
	}

创建会话:

	session = tg.Session()

训练:

	for step in range(100):
		loss_value = session.run(loss, feed_dict)
		if step % 10 == 0:
			print("Step:", step, " Loss:", loss_value)
		session.run(optimizer, feed_dict)

检索模型参数:

	weightg_value = session.run(weigths)
	biases_value = session.run(biases)

更多信息，参看`examples` 目录下的例子.
