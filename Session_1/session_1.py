#Session 1 Exercise Answer
#Group No. 2/ Tim Bohren Thomas Rueegg
import keras.backend as K
import numpy as np

#Ex 1

# a = K.placeholder(shape=(5,))
# b = K.placeholder(shape=(5,))
# c = K.placeholder(shape=(5,))
# 
# sqr_tensor = a*a+b*b+2*b*c
# 
# SQR_function = K.function(inputs=[a,b,c], outputs=[sqr_tensor])


#Ex 2

# x= K.placeholder(shape=())
# 
# tanh_tensor = (K.exp(x)-K.exp(-x))/(K.exp(x)+K.exp(-x))
# grad_tanh = K.gradients(loss = tanh_tensor, variables =[x])
# 
# tanh_function = K.function(inputs=[x], outputs=[tanh_tensor,grad_tanh[0]])

#Ex 3
w = K.placeholder(shape=(2,))
x = K.placeholder(shape=(2,))
b = K.placeholder(shape=(1,))

w_tensor = w[0]*x[0]+w[1]*x[1]+b
f_tensor = 1/(1+K.exp(w_tensor))
grad = K.gradients(loss = f_tensor, variables =[w])

f_function = K.function(inputs=[x], outputs=[f_tensor] + grad)


