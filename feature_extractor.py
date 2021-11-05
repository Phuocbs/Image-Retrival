"""
Write by Quang Van
"""
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

class FeatureExtractor:
    def __init__(self):
        base_model = ResNet50(weights="weight_resnet50.h5", include_top=False)
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        self.model = tf.keras.Model(inputs=base_model.input, outputs=x)
        
    def extract(self, img):
        
        x = preprocess_input(img)  # Subtracting avg values for each pixel
        feature = self.model.predict(x)  # (1, 2048)
        return feature # Normalize