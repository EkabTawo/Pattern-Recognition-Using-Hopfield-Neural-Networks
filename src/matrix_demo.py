

#Importing the necessary libraries and packages in python.
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


#Assigning the number of  neurons as well as the rows and columns for the matrix.
rows = 2
cols = 2
N = rows * cols

nrows = []
ncols = []
    
for n in range(N):
    if n+1 <=2:
        nrows.append(1)
    else:
        nrows.append(-1)
    #print(nrows)

#Applying the Hebbian Learning based on the formula to devlope the matrix.    
half_length = len(nrows)// cols
first_half, second_half = nrows[:half_length], nrows[half_length:]
    
new_list = [first_half, second_half]
    
    
p1 = np.array(nrows).reshape(-1,rows)
print(p1)

p1 = p1.reshape(-1)
p1 = p1.reshape(N,1)

print("Transpose Function\n", p1*p1.T)
print("Identity Function\n", np.identity(N))

#Calculation of weights
W = p1*p1.T-np.identity(N)
print("Weight\n", W)

#Calculation of the weighted sum to get the sign of the neurons. 
x = np.array(nrows).reshape(-1,rows)
x = x.reshape(-1)
x = x.reshape(N,1)

z = np.zeros(N)
for i in range(N):
    for j in range(N):
        if j != i:
            z[i] += W[i,j]*x[j]

x = np.sign(z)
print("Weighted Sum\n", z)
print('Sign\n', x)



