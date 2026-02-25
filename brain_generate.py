import numpy as np
import scipy.ndimage
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x) 
def generate():
    from minst_test import x as X_large, y_one_hot
    import matplotlib.pyplot as plt
    weights_h = np.load("weights_h.npy")
    weights_o = np.load("weights_o.npy")
    bias_h = np.load("bias_h.npy")
    bias_o = np.load("bias_o.npy")
    digit = int(input("What number should beagle try to show? (0-9): "))
    digit_indices = np.where(np.argmax(y_one_hot, axis=1)==digit)[0]#shot the indexes where requested number has the label
    confideces = []#array showing confidences of the number given
    for idx in digit_indices[:500]:#loop to see all images of requested number
       test_image = X_large[idx:idx+1]
       hidden = sigmoid(np.dot(test_image,weights_h)+bias_h)
       output = sigmoid(np.dot(hidden, weights_o) + bias_o)
       confideces.append((output[0,digit],idx))
    confideces.sort(reverse=True)#sort by confidence
    fig, axes = plt.subplots(3,3,figsize = (8,8))
    fig.suptitle(f"Beagle's favorite {digit}s (by confidence)")
    for i,(conf,idx) in enumerate(confideces[:9]):#loop images with requested label
        ax = axes[i//3, i%3]
        ax.imshow(X_large[idx].reshape(28,28),cmap='gray')
        ax.set_title(f'{conf:.3f}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()#open window to show