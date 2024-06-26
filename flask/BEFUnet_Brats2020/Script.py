import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import h5py
import numpy as np
import torch
import tensorflow as tf
from keras.models import load_model
from flask import Flask, request, render_template, jsonify
from configs.BEFUnet_configs import get_BEFUnet_configs
from models.BEFUnet import BEFUnet
from prediction_h5 import predictor_h5
from skimage.transform import resize
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__, template_folder='./web/', static_folder='./web/static/')

# Config Model
cfg = get_BEFUnet_configs()
model_seg = BEFUnet(cfg, n_classes=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_seg.to(device)
print("model_seg Config Done!")
# Load Model Weights
weights_path = r'F:\College Materials\Level 4\Semester 8\Graduation project\flask\BEFUnet_Brats2020\weights\model_weights_brats_9_Adam.pth'
weights_dict = torch.load(weights_path, map_location=device)
model_seg.load_state_dict(weights_dict)
print("Segmentatino Model Loading Done!")
# Predictor Initialization
pred = predictor_h5(model_seg, (224, 224))

model_path = r'F:\College Materials\Level 4\Semester 8\Graduation project\flask\res_classification\model\Classification.h5'  # Replace with the actual path to your saved model
model_class = load_model(model_path)
model_class.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Classification Model Loading Done!")


# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        app.logger.error('No file part in the request')
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        app.logger.error('No selected file')
        return jsonify({'error': 'No selected file'})
    
    if file and file.filename.endswith('.h5'):
        # Read data
        data = h5py.File(file, 'r')
        img, label = data['image'][:], data['mask'][:]
        img, label = np.max(img, axis=-1), np.max(label, axis=-1)

        # Model Inference
        output = pred.forward(data).squeeze(0).squeeze(0)
        
        # Add batch and channel dimensions
        image = np.expand_dims(img, axis=0)  # Add batch dimension
        image = np.expand_dims(image, axis=-1)  # Add channel dimension for grayscale
        prediction = model_class.predict(image)
        
        


        # Save images
        img_path = os.path.join(app.static_folder, 'image.png')
        label_path = os.path.join(app.static_folder, 'label.png')
        output_path = os.path.join(app.static_folder, 'output.png')

        plt.imsave(img_path, img, cmap='magma')
        plt.imsave(label_path, label, cmap='magma')
        plt.imsave(output_path, output, cmap='magma')

        return jsonify({
            'original_image_path': '/static/image.png',
            'label_image_path': '/static/label.png',
            'segmented_image_path': '/static/output.png',
            'classificaion_label' : f"{str(prediction[0][0])}"
        })
    else:
        app.logger.error('Invalid file type')
        return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True)
