
# Model Training

file : train.py

Two input variables are required : 
- **data_csv** : Path to the Input images csv file. Default *"./label.txt"*
- **img_directory** : Path to the directory of the images in *data_csv*. Default *"./images"*

Run the script to train the Deep Learning  model on 80% of the images in **data_csv**. The best model --based on the validation F1-score-- is saved to file **"./best_model.h5"**. The plot with the training loss is saved to **"./loss_curve.png"**.

# Model Inference

file : inference.py

Two input variables are required.
- **input_path** : Path to the input image. Default *"./images/image_525.jpg"*
- **model_weights** : Path to model weights file. Default *"./best_model.h5"*

The function outputs the prediction of the model as a vector of multi-class predictions.

# Transfer Learning : Model Parameters and Model Loss

I use the binary cross-entropy as the loss function for training the model. I tested a custom loss function to balance the dataset by weighting each class inversely proportional to its size, but it did not perform as well on this data. Hence a simple *binary_crossentropy* loss function is used to train the model. Other model parameters tested were:
- *size of the fully connected layer* after resnet50 : After validating on the validation data 64 was found to be the best size for the layer.
- *best Model CheckPoint* - The best model was chosen to be the model with the best validation F1-score rather than the model with the best accuracy.

# Handling Missing Labels 

If we do not want to make use of the image data for the y-label, we can essentially treat this problem as a data imputation problem. Since we do want to make use of the examples with partially missing labels, we do not want to just remove the rows with missing labels. In this case I tested a Matrix Factorization approach with simple L2 regularization : 

$$ \arg\min_{P,Q} BinaryCrossEntropy(S | PQ) + \lambda_2 ||P||^2 + \lambda_3 ||Q||^2 $$

where S denotes the labels and P, and Q the user and the content representative vectors respectively [1]

This approach did not produce satisfactory results for the dataset, hence a simple mean value imputation is used to replace the data. With more time a few more approaches to try are:
1. https://proceedings.mlr.press/v32/yu14.pdf

2. https://personal.ntu.edu.sg/boan/papers/IEEETC20.pdf

These approaches are worth trying since the regularization part in the loss function are dependent on the features of the input, which can be extracted from the resnet50 model in case of image data.




[1] https://everdark.github.io/k9/notebooks/ml/matrix_factorization/matrix_factorization.nb.html#21_real_value_matrix_factorization