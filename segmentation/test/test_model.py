import tensorflow as tf
from model import UNet

def test_unet():
    # Create model
    model = UNet(input_size=(256, 256, 3))
    
    # Test with random input
    test_input = tf.random.normal((1, 256, 256, 3))
    test_output = model(test_input)
    
    # Print shapes
    print("Input shape:", test_input.shape)
    print("Output shape:", test_output.shape)
    
    # Print model summary
    model.model.summary()

if __name__ == "__main__":
    test_unet()
