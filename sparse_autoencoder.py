#-coding:utf-8-
#!/usr/bin/env python
import numpy
import random
import line

import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import theano
import theano.tensor as T

from network.network import Network
import util

S = 8 
BASE_NUM = 500
TIMES = 10000
batch_size = 600

class Autoencoder(object):
	def __init__(self, nnet, input, dataset=None, learning_rate=0.01, beta=0.0, sparsity=0.01, weight_decay=0.0):
		if len(dataset) < 2:
			print "Error dataset must contain tuple (train_data,train_target)"
		train_data, train_target = dataset

		target = T.matrix('y')

		square_error = T.mean(0.5*T.sum(T.pow(target - nnet.output, 2), axis=1))

		avg_activate = T.mean(nnet.hiddenLayer[0].output, axis=0)
		sparsity_penalty = beta*T.sum(T.mul(T.log(sparsity/avg_activate), sparsity) + T.mul(T.log((1-sparsity)/T.sub(1,avg_activate)), (1-sparsity)))

		regularization = 0.5*weight_decay*(T.sum(T.pow(nnet.params[0],2)) + T.sum(T.pow(nnet.params[2],2)))

		cost = square_error + sparsity_penalty + regularization
		
		gparams = [T.grad(cost, param) for param in nnet.params]

		new_params = [param - (learning_rate*gparam) for param, gparam in zip(nnet.params, gparams)]

		updates = [(param, new_param) for param, new_param in zip(nnet.params, new_params)]

		index = T.lscalar()
		self.train = theano.function(
			inputs=[index],
			outputs=cost,
			updates=updates,
			givens={
				input: train_data[index * batch_size: (index + 1) * batch_size],
				target: train_target[index * batch_size: (index + 1) * batch_size]
			}
		)

		self.cost = theano.function(
			inputs=[],
			outputs=cost,
			givens={ input: train_data, target: train_target }
		)

def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.array(data_x,dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.array(data_y,dtype=theano.config.floatX), borrow=borrow)
        return (shared_x, shared_y)

def randomLines():
	'''
	# 8 * 8
	img = numpy.zeros((64), dtype=theano.config.floatX)
	direction = 1 if random.random() > 0.5 else 0
	c = int(random.random() * 8)
	for m in range(8):
		if direction == 1:
			img[c * 8 + m] = 1.
		else:
			img[m * 8 + c] = 1.
	#result = img
	#result *= 255
	#implot = plt.imshow(result, cmap=cm.Greys_r, vmin=0, vmax=255)
	#implot.set_interpolation('nearest')
	#plt.show()
	'''
	size = S
	img = line.getImage(size)
	new_img = []
	for i in range(size):	
		for j in range(size):
			new_img.append(1 - img[i][j] / 255.)

	return new_img

import random
def genRandomLines(num):
	ret = []
	for i in xrange(num):
		img = randomLines()
		if img is not None:
			ret.append(img)
	return { 'data': ret }

def getFragments():
	img = util.getImg('test.jpg')
	ret = util.getMore(util.gray(img))
	r = []
	for one in ret:
		new_img = []
		r.append(one.reshape(len(one) * len(one[0])))
		#for i in range(len(one)):	
		#	for j in range(len(one[i])):
		#		new_img.append(one[i][j])
		#		r.append(new_img)
	print 'count:', len(r)
	return { 'data': r }
		

def train(input, data):
	# 1000   8 * 8
        ds = shared_dataset((data['data'], data['data']))

	n_batch = len(data['data'])/batch_size
	nnet = Network(input=input, layers=(S * S, BASE_NUM, S * S), bias=True, activation="sigmoid", output_activation="sigmoid")
	trainer = Autoencoder(nnet=nnet, input = input, dataset=ds, learning_rate=1.2, beta=3.0, sparsity=0.01, weight_decay=0.0001)

	print 'N Batch:', n_batch
	print "epochs 0 \t", trainer.cost()
	index_array = [i for i in xrange(n_batch)]
	last_cost = 100000000000.
	for i in range(TIMES):
		random.shuffle(index_array)
		for index in index_array[:8]:
			trainer.train(index)

		if (i+1) % 50 == 0:
			cost = trainer.cost()
			if cost > last_cost:
				pass
				#break
			last_cost = cost
			print "epochs", i+1, "\t", cost
	
	print 'save parameters to test.net'
	nnet.save('test.net')
	return nnet

def showPredict(nnet):
	# predict
	params = theano.function(inputs=[], outputs=nnet.params)
	print 'Params:', params()
	W = nnet.params[0]
	pin = T.matrix('pin')
	output = T.dot(pin, W)
	predict = theano.function([ pin ], output)
	test = numpy.asarray([ randomLines() ])
	#test = numpy.transpose(test)
	print '-------PREDICT-----------'
	print test
	result =  test.reshape(S, S) * 255
	implot = plt.imshow(result, cmap=cm.Greys_r, vmin=0, vmax=255)
	implot.set_interpolation('nearest')
	plt.show()

	result = predict(test)
	print '----------RESULT-----------'
	print result
	result = T.nnet.sigmoid((result + 0.3) * 255).eval()
	result = result[0]

	fig = plt.figure()
	ind = numpy.arange(len(result))
	
	plt.bar(ind, result, width = .35)
	plt.show()

def showNnet(nnet):
	# Visualize
	weight = nnet.params[0].get_value()
	weight = numpy.transpose(weight)
	weight = weight - weight.mean()
	size = numpy.absolute(weight).max()
	weight = weight/size
	weight = weight*0.8+0.5

	image_per_row = int(BASE_NUM ** 0.5)
	result = []
	index = 0
	row = []
	print nnet.params
	for face in weight:
		face = face.reshape(S, S)
		face = numpy.lib.pad(face, (1,1), 'constant', constant_values=(0,0))
		if len(row) == 0:
			row = face
		else:
			row = numpy.concatenate([row,face],axis=1)

		if index % image_per_row == image_per_row-1:
			if len(result) == 0:
				result = row
			else:
				result = numpy.concatenate([result,row])
			row = []
		index += 1
	
	result *= 255
	implot = plt.imshow(result, cmap=cm.Greys_r, vmin=0, vmax=255)
	implot.set_interpolation('nearest')
	plt.show()


def testTrain():
	input = T.matrix('input')
	#data = sio.loadmat('dataset/patch_images.mat')
	#data = genRandomLines(10000)
	data = getFragments()
	nnet = train(input, data)
	showNnet(nnet)	

def testPredict():
	input = T.matrix('input')
	nnet = Network.load('test.net', input)
	showPredict(nnet)
	

import sys
if __name__ == "__main__":
	if len(sys.argv) > 1 and sys.argv[1] == 't':
		testTrain()
	else:
		testPredict()



