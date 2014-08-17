Sebastian Raschka  
Last updated 09/08/2014

# General tips and advices

<br>

### Sections

- [A typical pattern classification workflow](#a-typical-pattern-classification-workflow)
- [Resampling of training and test datasets](#resampling-of-training-and-test-datasets)
<br>
<br>

### A typical pattern classification workflow

[[back to top](#sections)]

<br>
<hr> 
(1) question -> (2) input data -> (3) features -> (4) algorithm -> (5) parameters -> (6) evaluation 
 <hr>
<br>

- The question should be concrete and specific: "Can I specify flower species based on the dimensions of the leaves?".

- Usually: the more input data is collected the better.

- Try to consult a domain expert for selecting "good" features.

- The choice of a learning and prediction algorithm is typically less important than good feature selection: 
	- Hand, David J. 2006. “Classifier Technology and the Illusion of Progress.” Statistical Science 21 (1): 1–14. doi:10.1214/088342306000000060. 
[http://projecteuclid.org/euclid.ss/1149600839](http://projecteuclid.org/euclid.ss/1149600839)

- It is expected that the generalization error (error on the test data set, or any other new dataset that was not used for training/fitting the model) is larger than the resubstitution error (= the error on the training dataset). However a large difference between the two error rates is a strong indicator for overfitting.

<br>
<br>

### Resampling of training and test datasets

[[back to top](#sections)]

- A typical, suggested ratio is to split an input dataset into 60% training and 40% test dataset.

- Both the test and training datasets should be randomly sampled.

- A test dataset should only be used **exactly once**, otherwise (if an algorithm is trained on the training dataset, and the test set is used to evaluate the predictor  multiple times) the test dataset "becomes" a training dataset and can cause "overfitting".

- If a test dataset is to be used to evaluate the predictor multiple times, create a separate validation dataset beforehand (the validation dataset should also only be used once).

- If the input dataset is split into three datasets, the typically suggested ratio is 60% training, 20% test, and 20% validation dataset.

- The choice whether a separate validation dataset should be used depends on the size of the dataset: A validation dataset is typically only recommended for "reasonably" large datasets

- It is generally recommended to use cross-validation: the training set is split into subsets of training and test datasets so that the original test dataset is kept to evaluate the predictor only once at the very end.

- Choose a cross-validation method (random sampling, k-fold, or leave-one-out) appropriate to your data: continuous or categorical data, large or small sample sizes.

- A typical size for **k** in k-fold cross-validation is 10 since a larger number of folds increases the variance of the error estimates and the computational time as well; very small numbers of folds can cause very biased error estimates on the other hand.
<br>
<br>

### Pre-processing

[[back to top](#sections)]

- Pre-processing procedures of the input data can have a larger impact on the prediction performance than the choice of a classifier itself so it should be done carefully.

- Standardization (scaling features to unit variance) is a requirement for many machine learning algorithms for optimal performance, especially if variables are measured on different scales and to account for skewed data points.

- Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) are two examples of commonly used dimensionality reduction. The goal is to reduce computational costs, remove "uninformative" (zero-variance) features and while retaining discriminatory information. Typically, LDA tends to outperform PCA for supervised training tasks since it does not only choose the directions of maximum variance like PCA, but also tries to maximize the class-separability.  
However, [A.M. Martinez et al., 2001](#http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=908974) showed that PCA can outperform LDA if the number of samples per class is relatively small.

- If you perform standardization, PCA, LDA, and any other transformation technique on the training dataset, it is important to always use the same parameters on the test dataset: E.g., in the case of standardization, use the same mean and standard deviation that was used to scale the training dataset.

<br>
<br>

### Missing Data

- Many machine learning algorithms will fail if missing data in a dataset is not treated appropriately.

- In general resubstitution via k-nearest neighbor imputation is considered to be superior over resubstitution of missing data by the overall sample mean.

<br>
<br>


### A typical KDD workflow

A typical Knowledge Discovery in Databases (KDD) workflow:

**Input data -> Data Preprocessing<sup>1</sup> -> Data Mining -> Post-processing -> information retrieval**

<sup>1</sup>Feature selection, dimensionality reduction, normalization, data subsetting ...

[[back to top](#sections)]

<br>
<br>


### in progress ...
[[back to top](#sections)]