import torch
import torchvision
# data set https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
import torch.nn as nn
# graphs? builds and trains layers of the neural network (input, hidden and output)
import torch.nn.functional as f
# applies a 2D transposed convolution over an input image composed of several input planes
# extracts local sliding blocks from a batched input tensor; combines an array of sliding local blocks into a large containing tensor
import torch.optim as optim
# optimizers


n_epochs = 3
# number of loops
batchSizeTrain = 64
batchSizeTest = 1000
learningRate = 0.01
momentum = 0.5
logInterval = 10
randomSeed = 1
# random seed - setting a random seed for anything that uses random generation will allow experiments
# to be repeated

torch.backends.cudnn.enabled = False
# disables cuDNN - removes nondeterministic algorithms

torch.manual_seed(randomSeed)
# batch size 64 to train model and 1000 to test the dataset; 0.1307 and 0.3081 are global mean and
# standard deviation for the MNIST dataset

# using torchvision to crop and normalize data; also downloads mnist data?
# https://pytorch.org/docs/stable/data.html
trainLoader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('/files/', train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(0.1307, 0.3081)])), batch_size=batchSizeTest, shuffle=True)
# normalizing values improves the model's ability to recognize data; normalizing tensor by adding normalized values?, idk if data parallelism is used but if so each worker is being trained by batch size. note shuffle allows the trainer to feed different samples into the training loop
# data loader is a python iterable over a data set; can be looped

testLoader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('/files/', train=False, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(0.1307, 0.3081)])), batch_size=batchSizeTest, shuffle=True)
# this is to test samples from mnist; note train is false

# DataLoader does not send anything to the GPU; cuda() does that; num_workers is default 0, when it is greater than 0 then data is loaded into main process (cpu)
# **note don't call cuda inside Dataset getitem__() - https://stackoverflow.com/questions/53998282/how-does-the-number-of-workers-parameter-in-pytorch-dataloader-actually-work
# gpu isn't being used (cudnn is false), we omit num_workers; otherwise it uses subprocesses to load data or used pinned RAM (pin_memory) to speed up ram to GPU transfers


# examining examples - comment out if unnecessary
examples = enumerate(testLoader)
print(examples)
# enumerate counts the number of loops it seems
batch_idx, (example_data, example_targets) = next(examples)
# batch_idx is the batch index; example_data holds rgb values for each pixel of the image, example_target is the corresponding number


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

        self.fc1 = nn.Linear(320, 50)
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
        x = x.view(-1, 320)
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


# initializing the network and optimizer
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learningRate, momentum=momentum)
# note if GPU was used during training, network parameters should also be sent to the GPU (using something such as network.cuda())

# training the model
trainLosses = []
trainCounter = []
testLosses = []
testCounter = [i*len(trainLoader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(trainLoader):
        optimizer.zero_grad()
        # sets gradients to NONE or zeros; can improve performance and ultimately improves the training of the model
        output = network(data)
        loss = f.nll_loss(output, target)
        # loss function; calculates and "scores" the difference between the output and the ground truth label
        # this loss function is used only on models with the softmax function as an output activation layer
        # when NLL is minimized, output improves; the logarithm punishes the model for making the correct prediction with smaller probabilities and encouraged for making the prediction with
        # higher probabilities

        loss.backward()
        # collects new gradients (weights and biases) to propagate back into the nn
        optimizer.step()
        # values are placed back into the network parameters using this

        if batch_idx % logInterval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(trainLoader.dataset), 100. * batch_idx / len(trainLoader), loss.item()))
            trainLosses.append(loss.item())
            # item() extracts the loss's value as a Python float
            trainCounter.append((batch_idx*64) + ((epoch-1)*len(trainLoader.dataset)))
            torch.save(network.state_dict(), '/results/model.pth')
            torch.save(optimizer.state_dict(), '/results/optimizer.pth')
            # state_dict() maps each layer to its parameter tensor
            # torch.save saves the network after each iteration
            # note: to access previous states, you can use .load_state_dict(state_dict)


def test():
    network.eval()
    # evaluates layers; is a switch for specific layers that behave differently during training and interference time
    # i.e. dropout and normalization layers need to be turned off during model evaluation, and .eval does it automatically
    testLoss = 0
    correct = 0
    with torch.no_grad():
        # neat - 'with' allows proper acquisition and release of resources
        for data, target in testLoader:
            output = network(data)
            testLoss += f.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            # i think it returns a tensore with the same dimension as the input but returns 1 max value for a prediction?
            correct += pred.eq(target.data.view_as(pred)).sum()
            #
            testLoss /= len(testLoader.dataset)
            testLosses.append(testLoss)
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(testLoss, correct, len(testLoader.dataset), 100. * correct / len(testLoader.dataset)))


# testing
test()
for epoch in range (1, n_epochs + 1):
    train(epoch)
    test()
