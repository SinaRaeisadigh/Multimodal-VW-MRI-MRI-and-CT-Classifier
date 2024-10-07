# Multimodal-VW-MRI-MRI-and-CT-Classifier

This project involves building a deep learning model to classify medical images using a multimodal approach. The images include Vessel Wall MRIs (3D), regular 2D MRIs, and CT scans, and the goal is to integrate these three modalities into a single model for improved classification performance. 

## (As I didn't have access to the datasets, I wrote the code without running it.)

Here's a detailed breakdown of the project:

### 1. **Data Preparation:**
   - The project starts with reading and organizing image data from three different directories: Vessel Wall MRI (VW MRI), regular MRI, and CT scans.
   - A CSV file contains the filenames of the images and their corresponding labels (the class each image belongs to).
   - Three lists are created to store the paths to the images for each modality: VW MRI, regular MRI, and CT scans.
   - The data is stored in a dictionary format, where each entry has paths to the three types of images and their associated label.

### 2. **Transforms:**
   - **3D VW MRI Transform**: A series of transformations are applied to the Vessel Wall MRI data, including adding a channel dimension, scaling intensity values, resizing the image to a fixed size (128x128x64), and converting it to a tensor.
   - **2D MRI & CT Transform**: Similar transformations are applied to the 2D MRI and CT images, but the spatial size is reduced to 128x128 since these are 2D images.

### 3. **Multimodal Dataset:**
   - A custom dataset class (`MultimodalDataset`) is created, which returns a sample consisting of three transformed images (VW MRI, regular MRI, and CT) along with their corresponding label.
   - The dataset is split into training and validation sets (80% training, 20% validation).

### 4. **Multimodal Neural Network:**
   - The model is a multimodal neural network, with separate DenseNet121 architectures handling each type of input:
     - **3D DenseNet121**: Processes the 3D VW MRI images.
     - **2D DenseNet121**: Processes the regular MRI and CT images.
   - The outputs of these three networks are concatenated into a single feature vector.
   - Fully connected layers (FC) further process this combined feature vector to make the final classification prediction.
   - Dropout layers are added between the FC layers to prevent overfitting.

### 5. **Training the Model:**
   - The model is trained using a cross-entropy loss function, which is commonly used for classification tasks.
   - The optimizer used is Adam with a learning rate of 1e-4, which adjusts the model's weights based on the gradients of the loss function.
   - During training, the model learns to predict the correct class for each set of inputs (VW MRI, regular MRI, and CT) by minimizing the loss.
   - After each epoch, the model is evaluated on a validation set, and metrics such as validation loss and accuracy are calculated to track the model's performance.

### 6. **Validation:**
   - After each training epoch, the model switches to evaluation mode and performs inference on the validation dataset.
   - It calculates the validation loss and accuracy, comparing predicted classes to the true labels.

