from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

app = Flask(__name__)

# Define the upload folder for user images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained style transfer model
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Function to apply style transfer to an image
def apply_style_transfer(content_image, style_image, content_weight=1e3, style_weight=1e-2):
    # Load and preprocess images
    content_image = tf.image.decode_image(tf.io.read_file(content_image))
    style_image = tf.image.decode_image(tf.io.read_file(style_image))
    
    # Stylize the content image
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    
    return stylized_image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'content_image' not in request.files or 'style_image' not in request.files:
            return redirect(request.url)
        
        content_image = request.files['content_image']
        style_image = request.files['style_image']
        
        if content_image.filename == '' or style_image.filename == '':
            return redirect(request.url)
        
        # Save uploaded images to the upload folder
        content_image_path = os.path.join(app.config['UPLOAD_FOLDER'], content_image.filename)
        style_image_path = os.path.join(app.config['UPLOAD_FOLDER'], style_image.filename)
        
        content_image.save(content_image_path)
        style_image.save(style_image_path)
        
        # Apply style transfer
        stylized_image = apply_style_transfer(content_image_path, style_image_path)
        
        # Save the stylized image
        stylized_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'stylized_image.jpg')
        tf.keras.utils.save_img(stylized_image_path, stylized_image.numpy())
        
        return render_template('result.html', stylized_image=stylized_image_path)
    
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
