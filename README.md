# Visual Taxomany

## 1. Introduction

### Brief Summary: [Kaggle Link](https://www.kaggle.com/competitions/visual-taxonomy)
In this competition, we aimed to predict product attributes (such as color, sleeve length, pattern, etc.) directly from product images. The key challenge lies in the discrepancy between product images and descriptions, which is common in e-commerce platforms. The goal was to develop an efficient model that can accurately predict attributes from images, saving time for sellers and ensuring a more accurate product listing process. Our approach involved training multiple deep learning models, including Vision Transformer, Swin Transformer, and ResNet, and combining their predictions through a custom Max Voting ensemble technique.

## 2. Data Preprocessing

### Data Preprocessing:
To prepare the data for model training, we performed several preprocessing steps:
- **Handling Missing Values**: Missing values in the dataset were handled using KNNImputer, which is effective in filling missing values based on the similarity between data points.
- **Outlier Detection and Removal**: Techniques were employed to detect and remove outliers from the dataset, though this was less of an issue given the structure of the data.

### Data Exploration and Learnings:
We explored the dataset to understand the distribution of product categories and attribute labels. Some attributes were highly imbalanced, requiring careful attention during model training to avoid bias towards more frequent classes.

### Training and Validation Data Split Strategy:
The data was split into training and validation sets using a stratified split to ensure that each set contained a representative distribution of categories and attribute labels. An 80/20 split was used for training and validation data.

### Feature Engineering:
No additional tabular features were engineered, as this is an image-based task. However, transformations such as image resizing, normalization, and data augmentation (random cropping and flipping) were applied to improve the model's generalization ability.

## 3. Modeling Approach

### Model Selection:
We experimented with three different architectures:
1. **Vision Transformer (ViT)**: A transformer-based model designed for image data, known for its effectiveness in capturing long-range dependencies in images.
2. **Swin Transformer**: A hierarchical vision transformer that works efficiently on images with varying scales and has become popular for vision tasks.
3. **ResNet**: A deep convolutional neural network known for its residual connections, which help alleviate the vanishing gradient problem and are highly effective in image classification tasks.

We selected these models because they represent cutting-edge architectures for image classification and are well-suited for the task of predicting multi-label attributes from product images.

### Architecture:
- **Vision Transformer (ViT)**: We used a pre-trained ViT model with fine-tuning on our dataset. The architecture includes multi-head self-attention layers and a feed-forward network, followed by a classification head.
- **Swin Transformer**: Similar to ViT, we used the pre-trained Swin Transformer model, which scales well with image sizes and captures both local and global features.
- **ResNet**: We used ResNet50, a deep CNN with 50 layers, which was fine-tuned for our specific task.

### Final Model and Hyperparameters:
- **Model Details**:
  - **ViT**: Used a 16x16 patch size and a depth of 12 layers.
  - **Swin Transformer**: Employed 12 layers and a window size of 7 for the sliding window attention mechanism.
  - **ResNet**: Implemented the ResNet50 architecture with batch normalization layers and ReLU activation.
- **Final Hyperparameters**:
  - **Learning Rate**: A learning rate of 1e-4 was used for all models after tuning on the validation set.
  - **Batch Size**: 32 images were used per batch.
  - **Epochs**: Each model was trained for 15 epochs with early stopping to avoid overfitting.
  - **Optimizer**: Adam optimizer was used with weight decay for regularization.

### Hyperparameter Fine-tuning:
We fine-tuned the models using a grid search approach on key hyperparameters like learning rate and batch size. Cross-validation was used to select the best combination.

## 4. Novelty and Innovation

### Unique Techniques:
- We used an **ensemble approach** by combining the outputs of the three models through Max Voting, which improves accuracy by leveraging the strengths of each model.
- The **early stopping strategy** was applied during training to prevent overfitting, which was crucial due to the high variance in the dataset.
- Additionally, we incorporated **data augmentation** to enhance the model's ability to generalize from product images with varying backgrounds, lighting, and angles.

## 5. Training Details

### Environment:
- **GPU/CPU**: We used an NVIDIA Tesla P100 GPU for model training with the following specifications: 16 GB RAM, 1.2 GHz clock speed.
- **Training Time**: Each model was trained for approximately 8 hours, with additional time required for hyperparameter tuning and ensemble training.
- **Training Time for Final Model**: The total time for training all three models and fine-tuning the ensemble was approximately 24 hours.

## 6. Evaluation Metrics

### Chosen Metrics:
We evaluated our models using the F1-score metric, specifically focusing on the **Macro F1-score**, which accounts for imbalanced data across the categories. We also used **Micro F1-score** to ensure good performance across all attributes.
- **Loss Function**: We used Binary Cross-Entropy Loss for multi-label classification tasks.

### Results:

| Category      | Micro-F1-score | Macro-F1-score |
|---------------|----------------|----------------|
| Men Tshirts   | 0.752          | 0.778          |
| Sarees        | 0.734          | 0.765          |
| Kurtis        | 0.758          | 0.749          |
| Women Tshirts | 0.765          | 0.740          |
| Women Tops    | 0.732          | 0.748          |

### Error Analysis:
Despite high accuracy, we observed that certain fine-grained attributes, such as distinguishing between different sleeve lengths (e.g., short-sleeve vs. long-sleeve), were challenging due to ambiguous images or mislabeled training data.

## 7. Conclusion

### Summary of Results:
Our final ensemble model, which combined Vision Transformer, Swin Transformer, and ResNet using Max Voting, achieved a public score of **0.76329** and a private score of **0.76242**. The performance was consistent across all product categories, with the model performing particularly well in categories like Sarees and Women Tops.

### Limitations and Future Improvements:
- **Limitations**: The model struggled with fine-grained attribute distinctions and sometimes made incorrect predictions for categories with more subtle differences.
- **Future Improvements**: To improve the model, we could explore the use of more advanced architectures such as Vision Transformers with hybrid models or employ additional data augmentation techniques to address class imbalance and improve fine-grained attribute predictions.

## 8. Appendix

### Additional Observations:
One interesting finding was the high correlation between color attributes and product category (e.g., dark colors for Men Tshirts and vibrant colors for Sarees).
