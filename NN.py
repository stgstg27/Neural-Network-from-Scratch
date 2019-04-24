import pandas as pd
import numpy as np
import math
import os
import imageio
import matplotlib.pyplot as plt
import random 

class Neural_Network():
	def __init__(self,hidden_layers, hidden_layers_neurons,input_dim):
		self.weight = [] #change this later
		self.img_Size = input_dim[0]*input_dim[1]
		self.hidden_layers = hidden_layers
		self.neurons_count = hidden_layers_neurons

	def image_preprocess(self,img):
		gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 
		gray = gray(img)
		gray = np.reshape(gray,(gray.shape[0]*gray.shape[1],1))
		gray/=255
		return gray


	def initialize(self):
		tmp_layers = [[random.randint(-100,100)/100.0 for i in range(self.img_Size)] for j in range(self.neurons_count[0])]
		self.weight.append(tmp_layers)
		for layer in range(self.hidden_layers)[1:]:
			tmp_layers = [[random.randint(-100,100)/100.0 for i in range(self.neurons_count[layer-1])] for j in range(self.neurons_count[layer])]
			self.weight.append(tmp_layers)

		tmp_layers = [[random.randint(-100,100)/100.0 for i in range(self.neurons_count[-1])] for j in range(2)]
		self.weight.append(tmp_layers)

	def compute_activation(self,val):
		return 1 / (1 + math.exp(-val))

	def feed_forward_network(self,input):

		flat_inp = self.image_preprocess(input)
		tmp_inp = flat_inp
		activation = []
		for layer in range(len(self.weight)):
			new_inp = [0]*len(self.weight[layer])
			for neuron in range(len(self.weight[layer])):
				new_inp[neuron] = np.dot(np.array(self.weight[layer])[neuron],tmp_inp)
				if type(new_inp[neuron]) == type([]):
					new_inp[neuron] = new_inp[neuron][0]
				new_inp[neuron] = self.compute_activation(new_inp[neuron]) 
				print "layer [" ,layer,"] -- "  , "neuron [" ,neuron,"]"," = " , new_inp[neuron]
			activation.append(new_inp)
			tmp_inp = new_inp

		print (tmp_inp)


if __name__=="__main__":
	pic = imageio.imread('1.jpg')
	nn = Neural_Network(2,[2,2],pic.shape)
	nn.initialize()
	nn.feed_forward_network(pic)