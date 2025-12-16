import torch
import onnx
from onnx2pytorch import ConvertModel
import tensorflow as tf
import tf2onnx

# Load the Keras model
keras_model = tf.keras.models.load_model('PredictWaste12.h5')

# Convert the Keras model to ONNX format
onnx_model_path = 'PredictWaste12.onnx'
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(keras_model, input_signature=spec, output_path=onnx_model_path)

# Load the ONNX model
onnx_model = onnx.load(onnx_model_path)

# Convert the ONNX model to a PyTorch model
pytorch_model = ConvertModel(onnx_model)

# Save the entire PyTorch model
torch.save(pytorch_model, 'PredictWaste12.pth')

print("Model conversion complete.")