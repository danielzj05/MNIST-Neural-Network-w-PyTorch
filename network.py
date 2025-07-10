import torch.nn as nn
# graphs? builds and trains layers of the neural network (input, hidden and output)

import torch.nn.functional as f
# applies a 2D transposed convolution over an input image composed of several input planes
# extracts local sliding blocks from a batched input tensor; combines an array of sliding local blocks into a large containing tensor
import torch.optim as optim
# optimizers


class Net(nn.Module):
    # torch.nn contains modules that help build nn models; will inherit useful methods
    def __init__(self):
        # init is always in a class and initializes the attributes of an object as soon as it is formed; it will always use self as an argument, representing the object of the class
        super(Net, self).__init__()
        # super function returns an object that represents the parent class; allows access to the parent class
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # conv1 is variable name with attributes from Net; nn.Conv2d applies a 2D convolution over an input signal composed of several input planes (tensors?)
        # first two arguments are in and out channels respectively; kernel size is the size of the convolving kernel or filter -- you can choose the size; 3 should be better, see:
        # https://www.sicara.fr/blog-technique/2019-10-31-convolutional-layer-convolution-kernel#:~:text=A%20common%20choice%20is%20to,%3A%203%2C%201%20by%20color.
        # input is ONE because rgb values are the same; grayscale

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.conv2_drop = nn.Dropout2d()
        # promotes independence between feature maps and is used instead of i.i.d dropout
        # zeroes out CHANNELS (feature maps); a portion of the output is lost
        # because size will change as well, the outputs are also scaled by 1/(1-p) (p for probability); this scaling allows the input and output to have a roughly equivalent mean: (see https://medium.com/@hunter-j-phillips/a-simple-introduction-to-dropout-3fd41916aaea#:~:text=Following%20this%20definition%2C%20PyTorch%27s%20nn,independently%20on%20every%20forward%20call)
        # *note that the probability is 0, meaning that the formula above isnt actually used?
        # note that because dropout occurs during training and not during interference time (actual utilisation)

        self.fc1 = nn.Linear(360, 50)
        # applies the linear transformation y = xA^T + B; fully connected layer *note you can use print(x.shape) to adapt the number of neurons in the linear layer (https://datascience.stackexchange.com/questions/47328/how-to-choose-the-number-of-output-channels-in-a-convolutional-layer)
        # will be used to classify the image into its label; reduces the number of layers
        # the input (320) must be equal to the size of the resulting tensor after the convolutions; the n-dimensional tensors are flattened to be classified (see Patrick Loeber pytorch tut.14)
        # output layers can be chosen at will

        self.fc2 = nn.Linear(50, 10)
        # this is the last layer; will identify a number between 0 and 9

    def forward(self, x):
        # pooling layers; shortening operations

        x = f.relu(f.max_pool2d(self.conv1(x), 2))
        # .relu is an activation function; decides whether a neuron should be activated or not (decides if part of the image is "recognized" or not; see 3blue1brown video, chap 1)
        # note: there are other variations such as ReLU, leaky ReLU, ELU and SiLU
        # USED to be sigmoid function or tanh (see stanford lectures)
        # for more see: https://builtin.com/machine-learning/relu-activation-function
        # new kernel size is 2x2; tensors are shrinking

        x = f.relu(f.max_pool2d(self.conv2_drop(x), 2))
        x = x.view(-1, 360)
        # different shape, same data as self; the -1 indicates for pytorch to calculate the number of rows for the tensor, while the columns remain specified: https://stackoverflow.com/questions/42479902/what-does-view-do-in-pytorch
        # essentially we are creating a tensor that can be flattened out later (i think)

        x = f.relu(self.fc1(x))
        x = f.dropout(x, training=self.training)
        # prevents over fitting/under fitting? yes - randomly zeroes some elements of the input tensor using probability p (default 0.5) and samples from a Bernoulli distribution

        x = self.fc2(x)
        # final linear layer; should determine a number

        return f.log_softmax(x)
        # returns a tensor; computes the output and gradient correctly as an alternative to log(softmax(x)) (doing the two operations separately is numerically unstable and slower
        # same shape as input tensor; probabilities are replaced by its logarithms
        # softmax(x) is a formula (see: https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax)
        # converts a vector of numbers into a probability distribution; log is used because it has better numerical stability and is less prone to under/overflow errors

