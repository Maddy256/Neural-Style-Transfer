# Neural-Style-Transfer
 ## REPORT
 ## INTRODUCTION: 
Neural Style Transfer (NST) is a cutting-edge technique in deep learning that allows for the transformation of an ordinary photograph into a masterpiece by 
applying the artistic style of renowned painters such as Van Gogh, Picasso, or Monet. This 
project aims to develop a neural style transfer model that leverages state-of-the-art deep 
learning techniques to merge the content of one image with the style of another, creating 
stunning visuals that seamlessly blend the two.
 ## OBJECTIVES:
 1. Extracting stylistic features from a given artwork.
 2. Applying these stylistic features to a different image while preserving the original content's 
   structure and details.
 3. Achieving a balance between the content and style to create visually appealing and artistically 
   coherent images.
 ## Techniques Used:
 This project explores techniques such as convolutional neural networks (CNNs), optimization 
algorithms, and perceptual loss functions. The model utilizes the VGG19 network, pre-trained 
on the ImageNet dataset, to extract features from the content and style images. The 
implementation avoids using external APIs and showcases the model through a web interface 
built with Streamlit.
 ## Methodology:
 ### 1. Environment Setup 
 To run the neural style transfer model, ensure the necessary libraries are installed:
 pip install tensorflow streamlit numpy pillow matplotlib
 ### 2. Loading and Preprocessing Images 
 The first step involves loading and preprocessing the content and style images. The 
 images are resized and normalized to facilitate processing by the VGG19 model.
 ### 3. Feature Extraction with VGG19
 The VGG19 model, pre-trained on the ImageNet dataset, is used to extract features 
 from the content and style images. Specific layers of the VGG19 model are chosen to 
 capture content and style representations.
 ### 4. Defining the Style and Content Loss
 The style loss is computed using the Gram matrix of the style image, which captures the 
 correlations between different feature maps. The content loss is computed as the mean 
 squared difference between the feature maps of the content image and the generated 
 image.
 ### 5. Optimization Algorithm
 An optimization algorithm is employed to iteratively update the generated image to 
 minimize the combined style and content loss. The Adam optimizer is used for this 
 purpose.
 ### 6. Combining Content and Style
 The generated image is updated to balance the content and style features, producing a 
 new image that retains the structure of the content image while adopting the style of the 
 style image.
 ### 7. Streamlit Web Interface
 A user-friendly web interface is created using Streamlit, allowing users to upload 
 content and style images and view the generated stylized image
 ## Significance of Content and Style Loss:
 ## Content Loss 
**Definition:** Content loss measures how well the generated image preserves the content 
of the original content image. It is typically defined as the Mean Squared Error (MSE) between the feature representations of the content image and the generated image, 
extracted from a particular layer of a Convolutional Neural Network (CNN)
 ## Significance:
 * **Content Preservation:** The main role of content loss is to ensure that the generated 
image keeps the semantic information of the content image. 
* **Feature Representation:** By comparing the feature representations from deep layers of 
VGG-19, content loss focus on higher-level content details rather than pixel-wise 
differences. 
* **Balancing:** In the NST process, content loss is balanced with style loss to achieve a 
mixup. The weight given to content loss determines how much the generated image will 
match the content image in terms of structure. 
## Style Loss:
 **Definition:** Style loss measures how well the generated image captures the style of 
the style image. It is computed by comparing the Gram matrices of the feature maps 
from different layers of a CNN. The Gram matrix captures the correlations between 
different features, representing the texture and style.
## Significance:
* **Texture and Patterns:** The primary role of style loss is to ensure that the generated 
image adopts the textures, colour, and patterns of the style image. 
* **Feature Correlations:** By using the Gram matrix, style loss captures the combination of 
features across the image, which is essential for representing the style.  
* **Layer-wise Contributions:** Style loss is usually computed at multiple layers of the CNN. 
Each layer contributes different levels of detail, from fine textures in lower layers to more 
abstract patterns in higher layers. This multi-scale approach helps in capturing a 
comprehensive style representation .
 *  **Balancing:** The weight assigned to style loss determines how much the generated image will adopt the style of the style image. 
# Code Implementation
## Importing Libraries
![Screenshot_3-7-2024_12128_](https://github.com/Maddy256/Neural-Style-Transfer/assets/118598865/4af2b915-5033-485e-a859-da1090ef6687)

  ## Setting up Streamlit

![Screenshot_3-7-2024_12229_](https://github.com/Maddy256/Neural-Style-Transfer/assets/118598865/5e8fc72b-932c-4dbb-9bdd-08bcf7e314cd)


## Helper Functions
Convert Tensor to Image

![Screenshot_3-7-2024_12243_](https://github.com/Maddy256/Neural-Style-Transfer/assets/118598865/b7e9fe9e-28bb-42c2-9052-772ee1818f53)


### Load and Preprocess Image

![Screenshot_3-7-2024_1239_](https://github.com/Maddy256/Neural-Style-Transfer/assets/118598865/844158ac-0514-43d2-a965-8d0338bdfa86)


### Display Image

![Screenshot_3-7-2024_12320_](https://github.com/Maddy256/Neural-Style-Transfer/assets/118598865/20be3676-b38a-4709-8e86-4a4d37b06dcb)


### Uploading Images

![Screenshot_3-7-2024_12334_](https://github.com/Maddy256/Neural-Style-Transfer/assets/118598865/3f5a4518-30bc-4638-af91-2761136913bd)


### Define Layers for Style and Content

![Screenshot_3-7-2024_12353_](https://github.com/Maddy256/Neural-Style-Transfer/assets/118598865/547b562d-db3a-4f53-a203-15e64193c296)
 
 
### Load VGG19 Model

![Screenshot_3-7-2024_1249_](https://github.com/Maddy256/Neural-Style-Transfer/assets/118598865/220a89e9-c2a3-4f60-b295-f839296897c2)


### Custom Model to Extract Style and Content

![Screenshot_3-7-2024_12426_](https://github.com/Maddy256/Neural-Style-Transfer/assets/118598865/67888d5f-842d-4188-b4c1-1a99f394ee86)


![Screenshot_3-7-2024_12439_](https://github.com/Maddy256/Neural-Style-Transfer/assets/118598865/bb4364dc-030c-482f-8eb0-ca73beba427c)


### Extracting Style and Content

![Screenshot_3-7-2024_12450_](https://github.com/Maddy256/Neural-Style-Transfer/assets/118598865/ccbeebee-eb12-46d4-bd18-706d552cd63d)


### Define Style and Content Loss Functions

![Screenshot_3-7-2024_1252_](https://github.com/Maddy256/Neural-Style-Transfer/assets/118598865/ec89e157-aab6-418b-9c7d-51754d8839d2)


### Training Configuration

![Screenshot_3-7-2024_12515_](https://github.com/Maddy256/Neural-Style-Transfer/assets/118598865/5e50fdf0-3ea0-45db-b781-389913402d6c)


### Training Step Function

![Screenshot_3-7-2024_12530_](https://github.com/Maddy256/Neural-Style-Transfer/assets/118598865/57f12f25-5b8e-4a34-94ba-833c16cb3733)


### Training Loop

![Screenshot_3-7-2024_12551_](https://github.com/Maddy256/Neural-Style-Transfer/assets/118598865/b6af4f45-5e6c-47af-bd14-fcf5b8ffb075)


### Saving and Displaying the Final Stylized Image

![Screenshot_3-7-2024_12631_](https://github.com/Maddy256/Neural-Style-Transfer/assets/118598865/149c90d7-2e67-465e-a8ed-f30ba5daf8af)



 # Results 
## Generated Images 
The results of the neural style transfer are evaluated based on 
visual inspection.

# Discussion 
Analysis of Results
 The model combined the style of one image with the content of 
another. The generated images exhibit the artistic patterns of the 
style image while preserving the spatial structure of the content 
image. 
# Significance 
* Artistic Applications: The ability to generate artwork by 
combining different styles and content. 
* Educational Use: Demonstrates the application of convolutional 
neural networks (CNNs) in image processing











