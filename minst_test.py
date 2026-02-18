from sklearn.datasets import fetch_openml
import numpy as  np
#downloading thw data
print("Downloading MNIST..(give me a minute man!)")
# Change line 5 in minst_test.py to this:
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
#normalizing pixels
x = mnist.data/255.0
y = mnist.target.astype(int)#what the files numbers actually are
#output neurons (what number the network guesses)
y_one_hot = np.zeros((y.size, 10))#create 70,000 rows and 10 colums, matrix computation is done here
y_one_hot[np.arange(y.size),y] =  1 #goes through each row and places a 1 in the column that matches the digit
print(f"Data Loaded! I believe this shape is: {x.shape}")
