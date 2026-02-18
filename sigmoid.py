from minst_test import x,y_one_hot
from minst_test import x as X_large, y_one_hot
import numpy as np;
#simple xor neural network
def sigmoid(x):
    return 1/(1+np.exp(-x)) #takes any number (from negative inf to postive inf and squishes it into a range between 0 and 1 for statistical reasons, it also mimics a biological neuron being off and on)

def sigmoid_derivative(x):
    return x * (1-x) #slope finder aka the senstivity of the iutput is to a change in the input, if neuron is alredy very certain, then the smaller the deriviative, assume oppiste as well

#the DATAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA (at least the data a literal tadpole can think about at one time)

x = np.array([[0,0],[0,1],[1,0],[1,1]])

y = np.array([[0],[1],[1],[0]])
#@parameters
input_size = 784
hiden_size = 16
output_size = 10
learning_rate = 0.5
epochs = 10000
#memory weights
weights_h = np.random.uniform(size=(input_size,hiden_size))*(np.sqrt(1./input_size))#squifification
weights_o = np.random.uniform(size=(hiden_size,output_size))*(np.sqrt(1./input_size))#squisification
#threshold, the number added to the sum to make sure neuron responds when the confidence or input is strong enough
bias_h = np.zeros((1,hiden_size))
bias_o = np.zeros((1,output_size))
#logic but also the training loop, which is Wa+b (weight,neuron and bias at the end)
for e in range(epochs):
    for i in range(0, 70000, 32):
        x_batch = X_large[i:i+32]
        y_batch = y_one_hot[i:i+32]
        hidden_layer_input = np.dot(x_batch,weights_h)+bias_h
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output,weights_o) + bias_o
        predicated_output = sigmoid(output_layer_input)
        #backward pass which it will learn from its mistakes
        error = y_batch - predicated_output
        #chain rule
        d_predicted_output = error * sigmoid_derivative(predicated_output)
        error_hidden_layer = d_predicted_output.dot(weights_o.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
        #nudgeing man
        weights_o += (hidden_layer_output.T.dot(d_predicted_output)/32)*learning_rate
        weights_h += (x_batch.T.dot(d_hidden_layer)/32) * learning_rate
        bias_o += (np.sum(d_predicted_output, axis=0, keepdims=True)/32) * learning_rate
        bias_h += (np.sum(d_hidden_layer, axis=0, keepdims=True)/32) * learning_rate
    predictions = np.argmax(predicated_output, axis =1)
    true_tables = np.argmax(y_batch, axis =1)
    accuracy = np.mean(predictions == true_tables)
    print(f"Epoch {e} complete I guess. Batch Accuracy: { accuracy}")
    if(e>100):
        break
print("Saving the brain...")
np.save('weights_h.npy', weights_h)
np.save('weights_o.npy', weights_o)
np.save('bias_h.npy', bias_h)
np.save('bias_o.npy', bias_o)
print("Brain saved! You can now turn off the computer without forgetting.")