import streamlit as st
import tensorflow as tf
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import time

# Set up Streamlit
st.title("Neural Style Transfer")

# Function to convert tensor to image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Function to load and preprocess image
def load_img(uploaded_file):
    max_dim = 512
    img = tf.image.decode_image(uploaded_file, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    scale = max_dim / max(shape)
    new_shape = tf.cast(shape * scale, tf.int32)
    
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# Function to display image
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.show()

# Upload images
content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

if content_file and style_file:
    content_image = load_img(content_file.read())
    style_image = load_img(style_file.read())

    st.image(tensor_to_image(content_image), caption='Content Image')
    st.image(tensor_to_image(style_image), caption='Style Image')

    # Define layers for style and content
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

    def get_vgg_layers(layer_names):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in layer_names]
        return tf.keras.Model([vgg.input], outputs)

    # Custom model to extract style and content
    class StyleContentModel(tf.keras.models.Model):
        def _init_(self, style_layers, content_layers):
            super(StyleContentModel, self)._init_()
            self.vgg = get_vgg_layers(style_layers + content_layers)
            self.style_layers = style_layers
            self.content_layers = content_layers
            self.num_style_layers = len(style_layers)
            self.vgg.trainable = False

        def call(self, inputs):
            inputs = inputs * 255.0
            preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
            outputs = self.vgg(preprocessed_input)
            style_outputs, content_outputs = outputs[:self.num_style_layers], outputs[self.num_style_layers:]
            style_outputs = [self.gram_matrix(style_output) for style_output in style_outputs]
            content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
            style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
            return {'content': content_dict, 'style': style_dict}

        @staticmethod
        def gram_matrix(input_tensor):
            result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
            num_locations = tf.shape(input_tensor)[1] * tf.shape(input_tensor)[2]
            return result / tf.cast(num_locations, tf.float32)

    if content_file and style_file:
        extractor = StyleContentModel(style_layers, content_layers)
        style_targets = extractor(style_image)['style']
        content_targets = extractor(content_image)['content']

        # Style and content loss functions
        def compute_loss(outputs, style_targets, content_targets, style_weight, content_weight, num_style_layers, num_content_layers):
            style_outputs = outputs['style']
            content_outputs = outputs['content']
            style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()])
            style_loss *= style_weight / num_style_layers
            content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) for name in content_outputs.keys()])
            content_loss *= content_weight / num_content_layers
            return style_loss + content_loss

        # Training configuration
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        style_weight = 1e-2
        content_weight = 1e4
        total_variation_weight = 30
        image = tf.Variable(content_image)

        # Training step function
        @tf.function
        def train_step(image, extractor, style_targets, content_targets, style_weight, content_weight, num_style_layers, num_content_layers, total_variation_weight):
            with tf.GradientTape() as tape:
                outputs = extractor(image)
                loss = compute_loss(outputs, style_targets, content_targets, style_weight, content_weight, num_style_layers, num_content_layers)
                loss += total_variation_weight * tf.image.total_variation(image)
            grad = tape.gradient(loss, image)
            optimizer.apply_gradients([(grad, image)])
            image.assign(tf.clip_by_value(image, 0.0, 1.0))

        # Training loop
        epochs = st.slider("Epochs", 1, 20, 5)
        steps_per_epoch = st.slider("Steps per Epoch", 50, 200, 100)

        start_time = time.time()

        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                train_step(image, extractor, style_targets, content_targets, style_weight, content_weight, len(style_layers), len(content_layers), total_variation_weight)
                st.text(f"Epoch {epoch + 1}, Step {step + 1}/{steps_per_epoch}")
                st.image(tensor_to_image(image), caption=f'Step {step + 1}', use_column_width=True)

        total_time = time.time() - start_time
        st.write(f"Total time: {total_time:.1f} seconds")

        # Save and display the final stylized image
        final_image = tensor_to_image(image)
        final_image.save('stylized-image.png')
        st.image(final_image, caption='Final Stylized Image')

        st.download_button(label="Download Image", data=open('stylized-image.png', 'rb'), file_name='stylized-image.png', mime='image/png')