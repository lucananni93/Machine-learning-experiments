import numpy as np
np.random.seed(42)

BINARY_DIM = 8
LARGEST_NUMBER = pow(2, BINARY_DIM)
# input variables
LR = 0.1
INPUT_DIM = 2
HIDDEN_DIM = 16
OUTPUT_DIM = 1

N_EPOCHS = 10000


def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output


# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)


def generate_binary_dataset() -> dict:
    int2binary = {}
    binary = np.unpackbits(
        np.array([range(LARGEST_NUMBER)], dtype=np.uint8).T, axis=1)
    for i in range(LARGEST_NUMBER):
        int2binary[i] = binary[i]
    return int2binary


def main():
    int2binary = generate_binary_dataset()

    # neural network weights
    # INPUT -> HIDDEN
    W_input_hidden = 2*np.random.random((INPUT_DIM, HIDDEN_DIM)) - 1
    # HIDDEN -> OUTPUT
    W_hidden_out = 2*np.random.random((HIDDEN_DIM, OUTPUT_DIM)) - 1
    # HIDDEN -> HIDDEN
    W_hidden_hidden = 2*np.random.random((HIDDEN_DIM, HIDDEN_DIM)) - 1

    # we save also the updates to give to each weight matrix
    W_input_hidden_update = np.zeros_like(W_input_hidden)
    W_hidden_out_update = np.zeros_like(W_hidden_out)
    W_hidden_hidden_update = np.zeros_like(W_hidden_hidden)

    for j in range(N_EPOCHS):
        # generate the first number and convert to binary
        a_int = np.random.randint(LARGEST_NUMBER / 2)
        a = int2binary[a_int]
        # generate the second number and convert to binary
        b_int = np.random.randint(LARGEST_NUMBER / 2)
        b = int2binary[b_int]
        # ground truth of the summation, and convert to binary
        c_int = a_int + b_int
        c = int2binary[c_int]
        # where we will store our best guess (binary encoded)
        d = np.zeros_like(c)

        overallError = 0

        # These two lists will keep track of the layer 2 derivatives and layer 1 values at each time step.
        layer_2_deltas = list()
        layer_1_values = list()
        # Time step zero has no previous hidden layer, so we initialize one that's off
        layer_1_values.append(np.zeros(HIDDEN_DIM))

        # moving along the positions of the binary encoding
        for position in range(BINARY_DIM):
            # input of the NN: the two bits at position BINARY_DIM - position - 1
            X = np.array([[a[BINARY_DIM - position - 1], b[BINARY_DIM - position - 1]]])
            # real output
            y = np.array([[c[BINARY_DIM - position - 1]]]).T

            # First, we propagate from the input to the hidden layer, Then, we propagate
            # from the previous hidden layer to the current hidden layer. Then we sum these
            # two vectors and pass through the sigmoid function.
            layer_1 = sigmoid(np.dot(X, W_input_hidden) + np.dot(layer_1_values[-1], W_hidden_hidden))
            # Propagate from the hidden layer to the output layer
            layer_2 = sigmoid(np.dot(layer_1, W_hidden_out))

            # calculate the error
            layer_2_error = y - layer_2
            # We're going to store the derivative in a list, holding the derivative at each time step.
            layer_2_deltas.append(layer_2_error * sigmoid_output_to_derivative(layer_2))
            # Calculate the sum of the absolute errors so that we have a scalar error (to track propagation).
            # We'll end up with a sum of the error at each binary position.
            overallError += np.abs(layer_2_error[0])

            # decode estimate so we can print it out
            d[BINARY_DIM - position - 1] = np.round(layer_2[0][0])

            # store hidden layer so we can use it in the next time step
            layer_1_values.append(np.copy(layer_1))

        future_layer_1_delta = np.zeros(HIDDEN_DIM)

        # So, we've done all the forward propagating for all the time steps, and we've computed
        # the derivatives at the output layers and stored them in a list. Now we need to backpropagate,
        # starting with the last time step, backpropagating to the first
        for position in range(BINARY_DIM):
            X = np.array([[a[position], b[position]]])
            # Selecting the current hidden layer from the list.
            layer_1 = layer_1_values[-position - 1]
            # Selecting the previous hidden layer from the list
            prev_layer_1 = layer_1_values[-position - 2]
            # Selecting the current output error from the list
            layer_2_delta = layer_2_deltas[-position - 1]
            # this computes the current hidden layer error given the error at the hidden layer from the future
            # and the error at the current output layer.
            layer_1_delta = (future_layer_1_delta.dot(W_hidden_hidden.T) +
                             layer_2_delta.dot(W_hidden_out.T)) * sigmoid_output_to_derivative(layer_1)
            # Now that we have the derivatives backpropagated at this current time step, we can construct our
            # weight updates (but not actually update the weights just yet). We don't actually update our weight
            # matrices until after we've fully backpropagated everything. Why? Well, we use the weight matrices
            # for the backpropagation. Thus, we don't want to go changing them yet until the actual backprop is done.
            W_hidden_out_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
            W_hidden_hidden_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
            W_input_hidden_update += X.T.dot(layer_1_delta)

            future_layer_1_delta = layer_1_delta

        W_input_hidden += W_input_hidden_update*LR
        W_hidden_hidden += W_hidden_hidden_update*LR
        W_hidden_out += W_hidden_out_update*LR

        W_input_hidden_update *= 0
        W_hidden_hidden_update *= 0
        W_hidden_out_update *= 0

        if j % 1000 == 0:
            print("Error: {}\tPred: {}\tTrue: {}".format(overallError, d, c))


if __name__ == '__main__':
    main()
