import numpy as np

# input array 
X = np.array() #train data
# output array
Y=np.array() #train data
Y = np.reshape(Y,(1,#no of training examples))
X_flat = X/90.0
W = np.random.rand(3,1)*0.01
b = 0
m=1532
print(np.shape(X))
print(np.shape(Y))

#Activation Function
def activation_function(z):
	#s = np.tanh(z)
	s = 1 / (1 + (np.exp(-z)))
	#leak = 0.2
	#f1 = 0.5 * (1 + leak)
	#f2 = 0.5 * (1 - leak)
	#s = f1 * z + f2 * abs(z)
	return s
   


epoch=2000 #Setting training iterations
lr=0.009#Setting learning rate

for i in range(epoch):

	#Forward Propogation
	Z = np.dot(W.T,X_flat) + b
	A = activation_function(Z)
	cost = -(1 / m) * np.sum(Y*np.log(A)+(1-Y)*np.log(1-A), axis = 1, keepdims = True)

	#Backpropagation
	dZ = Y-A
	dw = 1 / m * np.dot(X,(A-Y).T)
	db = 1 / m * np.sum(A-Y)
	W = W - lr * dw
	b = b - lr * db

M=np.array() #test data

M_flat = M/90.0

print(W,b)

out=np.array(activation_function(np.dot(W.T,M_flat)+b))

for d in np.nditer(out, op_flags=['readwrite']):
	print(d)

print(np.shape(out))
print(np.shape(Y))
print(cost)