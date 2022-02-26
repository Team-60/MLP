
class Layer:
    def forward(self):
        raise NotImplementedError
    def backward(self):
        raise NotImplementedError

if __name__ == "__main__":

    layer = Layer()

    layer.backward()
    layer.forward()
