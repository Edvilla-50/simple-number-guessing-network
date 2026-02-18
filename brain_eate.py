import numpy as np
from minst_test import x,y_one_hot

w_h = np.load('weights_h.npy')
w_o = np.load('weights_o.npy')
b_h = np.load('bias_h.npy')
b_o = np.load('bias_o.npy')

def sigmoid(x):
    return 1/(1+np.exp(-x))


test_index = np.random.randint(60000,70000)
test_image = x[test_index: test_index+1]
actual_label = np.argmax(y_one_hot[test_index])

hidden_layer = sigmoid(np.dot(test_image, w_h)+b_h)
output_layer = sigmoid(np.dot(hidden_layer, w_o)+b_o)

prediction = np.argmax(output_layer)#get the highetst index at the end of neural netowor
confidence = np.max(output_layer)*100 #turns decimal to percent
print(f"--- TEST RESULT ---")
print(f"I looked at image index #{test_index}")
print(f"Actual Number: {actual_label}")
print(f"Network's Guess: {prediction} ({confidence:.2f}% confident)")

if prediction == actual_label:
    print("The network got it right I'm so smart vro!")
else:
    print("The network was dumb dumb.")