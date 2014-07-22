Sebastian Raschka
last updated: 07/21/2014

# Data Science definitions in less than 50 words

<hr>
I would be happy to hear your comments and suggestions. 
Please feel free to drop me a note via
[twitter](https://twitter.com/rasbt), [email](mailto:bluewoodtree@gmail.com), or [google+](https://plus.google.com/+SebastianRaschka).
<hr>


<a class="mk-toclify" id="table-of-contents"></a>

#Table of Contents
- [Bayes Classifier](#bayes-classifier)
- [Curse of dimensionality](#curse-of-dimensionality)
- [Feature Selection](#feature-selection)
- [Kernel Density Estimation](#kernel-density-estimation)
- [Linear Discriminant Analysis (LDA)](#linear-discriminant-analysis-lda)
- [Maximum Likelihood Estimates (MLE)](#maximum-likelihood-estimates-mle)
- [Overfitting](#overfitting)
- [Parzen-Rosenblatt Window technique](#parzen-rosenblatt-window-technique)
- [Principal Component Analysis (PCA):](#principal-component-analysis-pca)
- [Support Vector Machines (SVM)](#support-vector-machines-svm)


# Data Science definitions in less than 50 words

<br>
<br>

<a class="mk-toclify" id="bayes-classifier"></a>
#### Bayes Classifier
[[back to top](#table-of-contents)]

A statistical pattern classification technique to classify objects based on Bayes' rule. The decision rule is based upon posterior probabilities that are calculated from the class-conditional probablities (likelihoods) and prior probabilities.

<br>
<br>

<a class="mk-toclify" id="curse-of-dimensionality"></a>
#### Curse of dimensionality
[[back to top](#table-of-contents)]

For a fixed number of training samples, the curse of dimensionality describes the increased error rate for a large number of dimensions (or features) due to imprecise parameter estimation and assumptions.

<br>
<br>

<a class="mk-toclify" id="feature-selection"></a>
#### Feature Selection 
[[back to top](#table-of-contents)]

An alternative algorithmic approach to projection-based techniques like Principal Component and Linear Discriminant Analysis for dimensionality reduction of a dataset in order to select a "sufficiently reduced" feature subset with minimal decline of the recognition rate of a classifier.

<br>
<br>

<a class="mk-toclify" id="kernel-density-estimation"></a>
#### Kernel Density Estimation
[[back to top](#table-of-contents)]

Non-parametric techniques (in constast to Maximum Likelihood Estimates) to estimate probability densities without requiring prior knowledge of the underlying model of the probability distribution.

<br>
<br>

<a class="mk-toclify" id="linear-discriminant-analysis-lda"></a>
#### Linear Discriminant Analysis (LDA)
[[back to top](#table-of-contents)]

A linear transformation technique (related to Principal Component Analysis) that is used to project a dataset onto a new feature space or feature subspace (for dimensionality reduction) where the new component axes maximize the spread between multiple classes. 

<br>
<br>

<a class="mk-toclify" id="maximum-likelihood-estimates-mle"></a>
#### Maximum Likelihood Estimates (MLE)

[[back to top](#table-of-contents)]

A parametric technique to estimate the values of class-conditional densities for a known statistical distribution or model. A popular example is the estimation of the "mean" and "variance" for a Gaussian distribution.

<br>
<br>

<a class="mk-toclify" id="overfitting"></a>
#### Overfitting
[[back to top](#table-of-contents)]

The result of using to many parameters that are necessary to describe a given model and thereby a detrimental increase of the complexity of the model.
A good indicator for overfitting is a high prediction accuracy on a training dataset but a low prediction accuracy on a test dataset.

<br>
<br>

<a class="mk-toclify" id="parzen-rosenblatt-window-technique"></a>
#### Parzen-Rosenblatt Window technique
[[back to top](#table-of-contents)]

A non-parametric kernel density estimation technique for probability densities of random variables if the underlying distribution/model is unknown. A so-called window function is used to count samples within hypercubes or Gaussian kernels of a specified volume to estimate the probability density.

<br>
<br>

<a class="mk-toclify" id="principal-component-analysis-pca"></a>
#### Principal Component Analysis (PCA):
[[back to top](#table-of-contents)]

A linear transformation technique that is commonly used to project a given dataset onto a new feature space  or feature subspace (for dimensionality reduction) where the new component axes are the directions that maximize the variance/spread of the data.

<br>
<br>

<a class="mk-toclify" id="support-vector-machines-svm"></a>
#### Support Vector Machines (SVM)
[[back to top](#table-of-contents)]

An algorithm that is used for supervised pattern classification tasks and regression analyses. The method is based on the creation of hyperplanes that are separating different classes in a dataset. 
