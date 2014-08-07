Sebastian Raschka  
last updated: 09/06/2014

# Terms in data science defined in less than 50 words

<br>

Although, I am not a big fan of the term "data science" itself  - since it is an umbrella term that basically includes everything but means nothing in particular.  
Here, I probably finally found a context where it is useful: I think it makes this collection eligible to include anything from machine learning and pattern classification, data mining, computer science, engineering, mathematics, and statistics. I basically picture "data science" as the name of the Venn diagram of all those different fields (with extensive overlap!).

<hr>

I'd be happy about contributions or to hear your comments and suggestions. 
Please feel free to drop me a note via
[twitter](https://twitter.com/rasbt), [email](mailto:bluewoodtree@gmail.com), or [google+](https://plus.google.com/+SebastianRaschka).

<hr>

<br>

<a class="mk-toclify" id="table-of-contents"></a>

#Table of Contents

- [Bagging](#bagging)
- [Confusion Matrix](#confusion-matrix)
- [Correlation analysis](#correlation-analysis)
- [Correlation analysis - Pearson](#correlation-analysis-pearson)
- [Correlation analysis - Spearman](#correlation-analysis-spearman)
- [Cosine Similarity](#cosine-similarity)
- [Cross-validation](#cross-validation)
- [Curse of dimensionality](#curse-of-dimensionality)
- [Data mining](#data-mining)
- [Decision rule](#decision-rule)
- [Decision tree classifier](#decision-tree-classifier)
- [Density-based clustering](#density-based-clustering)
- [Distance Metric Learning](#distance-metric-learning)
- [Eigenvectors and Eigenvalues](#eigenvectors-and-eigenvalues)
- [Ensemble methods](#ensemble-methods)
- [Feature Selection Algorithms](#feature-selection-algorithms)
- [Feature Space](#feature-space)
- [Fuzzy C-Means Clustering](#fuzzy-cmeans-clustering)
- [Independent Component Analysis](#independent-component-analysis)
- [Kernel Density Estimation](#kernel-density-estimation)
- [Kernel Methods](#kernel-methods)
- [K-fold cross-validation](#k-fold-cross-validation)
- [K-Means Clustering](#k-means-clustering)
- [K-Means++ Clustering](#k-means-clustering-1)
- [K-Medoids Clustering](#k-medoids-clustering)
- [LASSO Regression](#lasso-regression)
- [Least Squares fit](#least-squares-fit)
- [Least Squares fit - linear](#least-squares-fit-linear)
- [Linear Discriminant Analysis (LDA)](#linear-discriminant-analysis-lda)
- [Logistic Regression](#logistic-regression)
- [Machine learning](#machine-learning)
- [Mahalanobis distance](#mahalanobis-distance)
- [Markov chains](#markov-chains)
- [Monte Carlo simulation](#monte-carlo-simulation)
- [Naive Bayes Classifier](#naive-bayes-classifier)
- [Maximum Likelihood Estimates (MLE)](#maximum-likelihood-estimates-mle)
- [Normalization - Min-Max Scaling](#normalization-min-max-scaling)
- [Normalization - Standard Scores](#normalization-standard-scores)
- [Objective function](#objective-function)
- [Overfitting](#overfitting)
- [Parzen-Rosenblatt Window technique](#parzen-rosenblatt-window-technique)
- [Pattern classification](#pattern-classification)
- [Precision and Recall](#precision-recall)
- [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
- [Purity Measure](#purity-measure)
- [Random forest](#random-forest)
- [Reinforcement learning]	(#reinforcement-learning)
- [Regularization](#regularization)
- [Ridge Regression](#ridge-regression)
- [Singular Value Decomposition](#singular-value-decomposition)
- [Supervised learning](#supervised-learning)
- [Support Vector Machine (SVM)](#support-vector-machine)
- [Unsupervised learning](#unsupervised-learning)
- [Whitening transformation](#whitening-transformation)



<br>
<br>

<a class="mk-toclify" id="bagging"></a>
#### Bagging
[[back to top](#table-of-contents)]

Bagging is an ensemble method for classification (or regression analysis) in which individual models are trained by random sampling of data, and the final decision is made by voting among individual models with equal weights (or averaging for regression analysis).   
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>

<a class="mk-toclify" id="confusion-matrix"></a>
#### Confusion Matrix
[[back to top](#table-of-contents)]

The confusion matrix is used as a way to represent the performance of a classifier and is sometimes also called "error matrix". This square matrix consists of columns and rows that list the number of instances as absolute or relative "actual class" vs. "predicted class" ratios.

<br>
<br>

<a class="mk-toclify" id="correlation"></a>
#### Correlation analysis
[[back to top](#table-of-contents)]

Correlation analysis describes and quantifies the relationship between two independent variables. Typically, in case of a positive correlation both variables have a tendency to increase, and in the case of negative correlation, one variable increases while the other variable increases. It is important to mention the famous quotation "correlation does not imply causation".

<br>
<br>

<a class="mk-toclify" id="correlation-analysis-pearson"></a>
#### Correlation analysis - Pearson
[[back to top](#table-of-contents)]

The Pearson correlation coefficientis probably the most widely used measure for a linear relationship between two normal distributed variables and thus often just called "correlation coefficient". Usually, the Pearson coefficient is obtained via a [Least-Squares fit](#least-squares-fit) and a value of 1 represents a perfect positive relation-ship, -1 a perfect negative relationship, and 0 indicates the absence of a relationship between variables.

<br>
<br>

<a class="mk-toclify" id="correlation"></a>
#### Correlation analysis - Spearman
[[back to top](#table-of-contents)]

The Spearman correlation coefficient is a rank-based version of [Pearson's correlation coefficient](correlation-analysis-pearson), which can be used for variables that are not normal-distributed and have a non-linear relationship.

<br>
<br>

<a class="mk-toclify" id="cosine-similarity"></a>
#### Cosine Similarity
[[back to top](#table-of-contents)]

Cosine similarity measures the orientation of two *n*-dimensional sample vectors irrespective to their magnitude. It is calculated by the dot product of two numeric vectors, and it is normalized by the vector lengths, so that output values close to 1 indicate high similarity.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>

<a class="mk-toclify" id="cross-validation"></a>
#### Cross-validation
[[back to top](#table-of-contents)]

Cross-validation is a statistical technique to estimate the prediction error rate by splitting the data into training, cross-validation, and test sets. A prediction model is obtained using the training set, and model parameters are optimized by the cross-validation set, while the test set is held primarily for empirical error estimation.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>

<a class="mk-toclify" id="curse-of-dimensionality"></a>
#### Curse of dimensionality
[[back to top](#table-of-contents)]

For a fixed number of training samples, the curse of dimensionality describes the increased error rate for a large number of dimensions (or features) due to imprecise parameter estimations.

<br>
<br>

<a class="mk-toclify" id="data-mining"></a>
#### Data mining
[[back to top](#table-of-contents)]

A field that is closely related to machine learning and pattern classification. The focus of data mining is often on the collection of data and combines different techniques the data in order to extract meaningful information out of data.

<br>
<br>

<a class="mk-toclify" id="dbscan"></a>
#### DBSCAN
[[back to top](#table-of-contents)]

DBSCAN is a variant of a density-based clustering algorithm that identifies core points as regions of high-densities based on their number of neighbors (> *MinPts*) in a specified radius (&epsilon;). Points that are below *MinPts* but within &epsilon; are specified as border points; the remaining points are classified as noise points.   
([Vahid Mirjalili](https://github.com/mirjalil); edited)

<br>
<br>

<a class="mk-toclify" id="decision-rule"></a>
#### Decision rule
[[back to top](#table-of-contents)]

A function in pattern classification tasks of making an "action", e.g., assigning a certain class label to an observation or pattern.

<br>
<br>

<a class="mc-toclify" id="decision-tree"></a>
#### Decision tree classifier
[[back to top](#table-of-contents)]

Decision tree classifiers are tree like graphs, where nodes in the graph test certain conditions on a particular set of features, and branches split the decision towards the leaf nodes. Leaves represent lowest level in the graph and determine the class labels. Optimal tree are trained by minimizing Gini impurity, or maximizing information gain.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil); edited)


<br>
<br>


<a class="mc-toclify" id="density-dased-clustering"></a>
#### Density-based clustering
[[back to top](#table-of-contents)]

In density-based clustering, regions of high density in n-dimensional space are identified as clusters. The best advantage of this class of clustering algorithms is that they do not require apriori knowledge of number of clusters (as opposed to [k-means](#k-means-clustering) algorithm).   
(submitted by [Vahid Mirjalili](https://github.com/mirjalil); edited)

<br>
<br>

<a class="mc-toclify" id="distance-metric-learning"></a>
#### Distance Metric Learning
[[back to top](#table-of-contents)]

Distance metrics are fundamental for many machine learning algorithms. Distance metric learning - instead of learning a model - incorporates estimated relevances of features to obtain a distance metric for potentially optimal separation of classes and clusters: Large distances for objects from different classes, and small distances for objects of the same class, respectively.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil); edited)

<br>
<br>

<a class="mc-toclify" id="eigenvectors-and-eigenvalues"></a>
#### Eigenvectors and Eigenvalues
[[back to top](#table-of-contents)]

Both eigenvectors and eigenvalues fundamental in many applications involve linear systems and are related via *A&middot;v = &lambda;&middot;v* (where *A* is a square matrix, *v* the eigenvector, and &lambda; the eigenvalue). Eigenvectors are describing the direction of the axes of a linear transformation, whereas eigenvalues are describing the scale or magnitude.

<br>
<br>

<a class="mc-toclify" id="ensemble-methods"></a>
#### Ensemble methods
[[back to top](#table-of-contents)]

Ensemble methods combine multiple classifiers which may differ in algorithms, input features, or input samples. Statistical analyses showed that ensemble methods yield better classification performances and are also less prone to overfitting. Different methods, e.g., bagging or boosting, are used to construct the final classification decision based on weighted votes.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil); edited)

<br>
<br>

<a class="mk-toclify" id="feature-selection-algorithms"></a>
#### Feature Selection Algorithms
[[back to top](#table-of-contents)]

Algorithmic approaches as alternative to projection-based techniques like Principal Component and Linear Discriminant Analysis for dimensionality reduction of a dataset via the selection a "sufficiently reduced" feature subsets with minimal decline of the recognition rate of a classifier.

<br>
<br>

<a class="mk-toclify" id="feature-space"></a>
#### Feature Space
[[back to top](#table-of-contents)]

A feature space describes the descriptive variables  that are available for samples in a dataset as a *d*-dimensional Euclidean space. E.g., sepal length and width, and petal length and width for each flower sample in the popular Iris dataset.

<br>
<br>

<a class="mk-toclify" id="fuzzy-cmeans-clustering"></a>
#### Fuzzy C-Means Clustering
[[back to top](#table-of-contents)]

Fuzzy C-Means is a soft clustering algorithm in which each sample point has a membership degree to each cluster; in hard (crisp) clustering, membership of each point to each cluster is either 0 or 1. Fuzzy C-Means considers a weight matrix for cluster memberships, and minimizes sum squared error (SSE) of weighted distances of sample points to the cluster centroids.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>

<a class="mk-toclify" id="independent-component-analysis"></a>
#### Independent Component Analysis
[[back to top](#table-of-contents)]

Independent Component Analysis (ICA) is a statistical signal-processing technique that decomposes a multivariate dataset of mixed, non-gaussian distributed source signals into  independent components 
A popular example is the separation of overlapping voice samples -- the so-called "cocktail party problem".

<br>
<br>

<a class="mk-toclify" id="kernel-density-estimation"></a>
#### Kernel Density Estimation
[[back to top](#table-of-contents)]

Non-parametric techniques to estimate probability densities from the available data without requiring prior knowledge of the underlying model of the probability distribution.

<br>
<br>

<a class="mk-toclify" id="kernel-methods"></a>
#### Kernel Methods
[[back to top](#table-of-contents)]

Kernel methods are algorithms that map the sample vectors of a dataset onto a (typically) higher-dimensional feature space via a so-called kernel function. The goal is to identify  and simplify general relationships between data, which is especially useful for linearly non-separable datasets.  

<br>
<br>

<a class="mk-toclify" id="k-fold-cross-validation"></a>
#### k-fold Cross-validation
[[back to top](#table-of-contents)]

In k-fold cross-validation the data is split into *k* subsets, then a prediction/classification model is trained *k* times, each time holding one subset as test set, training the model parameters using the remaining *k*-1. Finally, cross-validation error is evaluated as the average error out of all *k* training models.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>

<a class="mk-toclify" id="k-means-clustering"></a>
#### K-Means Clustering
[[back to top](#table-of-contents)]

A method of partitioning a dataset into *k* clusters by picking *k* random initial points (where *k* < *n*, the number or total points - modified by *S.R.*), assigning clusters, averaging, reassigning, and repeating until stability is achieved. The number *k* must be chosen beforehand.  
(submitted by [Jack Maney](https://github.com/jackmaney))

<br>
<br>

<a class="mc-toclify" id="k-means-clustering-1"></a>
#### K-Means++ Clustering
[[back to top](#table-of-contents)]

A variant of [k-means](#k-means-clustering) where instead of choosing all initial centers randomly, the first is chosen randomly, the second chosen with probability proportional to the squared distance from the first, the third chosen with probability proportional to the square distance from the first two, etc. See [this paper](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf).   
(submitted by [Jack Maney](https://github.com/jackmaney))

<br>
<br>

<a class="mk-toclify" id="k-medoids-clustering"></a>
#### K-Medoids Clustering
[[back to top](#table-of-contents)]

K-Medoids clustering is a variant of [k-means](#k-means-clustering) algorithm in which cluster centroids are picked among the sample points rather than the mean point of each cluster. K-Medoids can overcome some of the limitations of [k-means](#k-means-clustering) algorithm by avoiding empty clusters, being more robust to outliers, and being more easily applicable to non-numeric data types.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>

<a class="mk-toclify" id="lasso-regression"></a>
#### LASSO Regression
[[back to top](#table-of-contents)]

LASSO (Least Absolute Shrinkage and Selection Operator) is a regression model that uses the L1-norm (sum of absolute values) of model coefficients to penalize the model complexity. LASSO has the advantage that some coefficients can become zero, as opposed to [ridge regression](#ridge-regression) that uses the squared sum of model coefficients.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>

<a class="mc-toclify" id="least-squares-fit"></a>
#### Least Squares fit
[[back to top](#table-of-contents)]

A regression technique to fit a linear or nonlinear model to a dataset to identify relationships between variables by minimizing the norm of residuals squared. 

<br>
<br>

<a class="mc-toclify" id="least-squares-fit-linear"></a>
#### Least Squares fit - linear
[[back to top](#table-of-contents)]

A linear regression technique that fits a straight line to a data set (or overdetermined system) by minimizing the sum of the squared residuals, which can be the minimized vertical or perpendicular offsets from the fitted line.

<br>
<br>

<a class="mk-toclify" id="linear-discriminant-analysis-lda"></a>
#### Linear Discriminant Analysis (LDA)
[[back to top](#table-of-contents)]

A linear transformation technique (related to Principal Component Analysis) that is commonly used to project a dataset onto a new feature space or feature subspace, where the new component axes maximize the spread between multiple classes, or for classification of data.

<br>
<br>

<a class="mk-toclify" id="logistic-regression"></a>
#### Logistic Regression
[[back to top](#table-of-contents)]

Logistic regression is a statistical model used for binary classification (binomial logistic regression) where class labels are mapped to "0" or "1" outputs. Logistic regression uses the logistic function (a general form of sigmoid function), where its output ranges from (0-1).

(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>

<a class="mk-toclify" id="machine-learning"></a>
#### Machine learning

[[back to top](#table-of-contents)]

A set of algorithmic instructions for discovering and learning patterns from data e.g., to train a classifier for a pattern classification task.

<br>
<br>

<a class="mk-toclify" id="mahalanobis-distance"></a>
#### Mahalanobis distance

[[back to top](#table-of-contents)]

The Mahalanobis distance measure accounts for the covariance among variables by calculating the distance between a sample *x* and the sample mean &mu; in units of the standard deviation.  The Mahalanobis distance becomes equal to the Euclidean distance for uncorrelated with same variances.

<br>
<br>

<a class="mk-toclify" id="markov-chains"></a>
#### Markov chains

[[back to top](#table-of-contents)]

Markov chains (names after Andrey Markov) are mathematical systems that describe the transitioning between different states in a model. The transitioning from one state to the other (or back to itself) is a stochastic process.

<br>
<br>

<a class="mk-toclify" id="monte-carlo-simulation"></a>
#### Monte Carlo simulation

[[back to top](#table-of-contents)]

A Monte Carlo simulation is an iterative sampling method for solving deterministic models. Random numbers or variables from a particular probability distribution are used as input variables for uncertain parameters to compute the response variables. 


<br>
<br>

<a class="mk-toclify" id="maximum-likelihood-estimates-mle"></a>
#### Maximum Likelihood Estimates (MLE)

[[back to top](#table-of-contents)]

A technique to estimate the parameters that have been fit to a model by maximizing a known likelihood function. One common application is the estimation of "mean" and "variance" for a Gaussian distribution.

<br>
<br>

<a class="mk-toclify" id="naive-bayes-classifier"></a>
#### Naive Bayes Classifier
[[back to top](#table-of-contents)]

A classifier based on a statistical model (i.e., Bayes theorem: calculating posterior probabilities based on the prior probability and the so-called likelihood) in the field of pattern classification. It is based on the assumption that all features are i.i.d. (independent and identically distributed). 

<br>
<br>

<a class="mk-toclify" id="objective-function"></a>
#### Objective function
[[back to top](#table-of-contents)]

A function that is to be optimized (minimizing or maximizing a numerical value depending on a particular task or problem), for example, an objective function in pattern classification tasks could be to minimize the error rate of a classifier.

<br>
<br>

<a class="mk-toclify" id="normalization-min-max-scaling"></a>
#### Normalization - Min-Max scaling
[[back to top](#table-of-contents)]

A data pre-processing step (also often referred to as "Feature Scaling") for fitting features from different measurements within a certain range, typically the unit range from 0 to 1.

<br>
<br>

<a class="mk-toclify" id="normalization-standard-scores"></a>
#### Normalization - Standard Scores
[[back to top](#table-of-contents)]

A data pre-processing step (also often just called "Standardization") for re-scaling features from different measurements to match proportions of a standard normal distribution (unit variance centered at mean=0).

<br>
<br>

<a class="mk-toclify" id="parzen-rosenblatt-window-technique"></a>
#### Parzen-Rosenblatt Window technique
[[back to top](#table-of-contents)]

A non-parametric kernel density estimation technique for probability densities of random variables if the underlying distribution/model is unknown. A so-called window function is used to count samples within hypercubes or Gaussian kernels of a specified volume to estimate the probability density.

<br>
<br>

<a class="mk-toclify" id="pattern-classification"></a>

#### Pattern classification

[[back to top](#table-of-contents)]

The usage of patterns in datasets to discriminate between classes, i.e., to assign a class label to a new observation based on inference.

<br>
<br>

<a class="mk-toclify" id="principal-component-analysis-pca"></a>
#### Principal Component Analysis (PCA)
[[back to top](#table-of-contents)]

A linear transformation technique that is commonly used to project a dataset (without utilizing class labels) onto a new feature space or feature subspace (for dimensionality reduction) where the new component axes are the directions that maximize the variance/spread of the data. 

<br>
<br>

<a class="mk-toclify" id="precision-recall"></a>
#### Precision and Recall
[[back to top](#table-of-contents)]

Precision and recall are two measures to assess performance of a classifier if class label distributions are skewed.
Precision is defined as the ratio of number of relevant items out of total retrieved items, whereas recall is the fraction of relevant items which are retrieved.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil); edited)

<br>
<br>

<a class="mk-toclify" id="purity-measure"></a>
#### Purity Measure

[[back to top](#table-of-contents)]

In a cluster analysis with given truth cluster memberships (or classes), "purity" is used to assess the effectiveness of clustering. Purity is measured by assigning each cluster to the class that is maximally represented and computed via the weighted average of maximum number of samples from the same class in each cluster.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>


<a class="mk-toclify" id="random-forest"></a>
#### Random forest

[[back to top](#table-of-contents)]

Random forest is an ensemble classifier where multiple [decision tree classifiers](#decision-tree-classifier) are learned and combined via the [bagging](#bagging) technique. Unseen/test objects are then classified by taking the majority of votes from individual decision trees.
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>



<a class="mk-toclify" id="regularization"></a>
#### Regularization

[[back to top](#table-of-contents)]

Regularization is a technique to overcome overfitting by introducing a penalty term for model complexity. Usually, the penalty term is the squared sum of the model parameters, thereby promoting less complex models during training. Regularization may increase the training error but can potentially reduce the classification error on the test dataset.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>


<a class="mk-toclify" id="reinforcement-learning"></a>
#### Reinforcement learning

[[back to top](#table-of-contents)]

Reinforcement learning is a machine learning algorithm that learns from a series of actions by maximizing a "reward function". The reward function can either be maximized by penalizing "bad actions" and/or rewarding "good actions".

<br>
<br>



<a class="mk-toclify" id="ridge-regression"></a>
#### Ridge Regression

[[back to top](#table-of-contents)]

Ridge regression is a regularized regression technique in which the squared sum of the model coefficients is used to penalize model complexity.   
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>


<a class="mk-toclify" id="singular-value-decomposition"></a>
#### Singular Value Decomposition

[[back to top](#table-of-contents)]

Singular value decomposition (SVD) is linear algebra technique that decomposes matrix X into   
***U D V<sup>T<sup>*** where ***U*** (left-singular vectors) and ***V*** (right-singular vector) are both column-orthogonal, and ***D*** is a diagonal matrix that contains singular values. [PCA](#principal-component-analysis) is closely related to the right-singular vectors of SVD.  
 (submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<a class="mk-toclify" id="supervised-learning"></a>
#### Supervised Learning

[[back to top](#table-of-contents)]

The problem of inferring a mapping between the input space X and a target variable y when given labelled training data (i.e. (X,y) pairs). Encompasses the problems of classification (categorical y) and regression (continuous y).  
(submitted by [Alex Yakubovich](https://github.com/ayakubovich))

<br>
<br>

<a class="mk-toclify" id="support-vector-machine"></a>
#### Support Vector Machine

[[back to top](#table-of-contents)]

SMV is a classification  method that tries to find the hyperplane which separates classes with highest margin. The margin is defined as the minimum distance from sample points to the hyperplane. The sample point(s) that form margin are called support vectors and eventually establish the SVM model.   
(submitted by [Vahid Mirjalili](https://github.com/mirjalil); edited)

<br>
<br>

<a class="mk-toclify" id="unsupervised-learning"></a>
#### Unsupervised Learning

[[back to top](#table-of-contents)]

The problem of inferring latent structure in data when not given any training cases. Encompasses the problems of clustering, dimensionality reduction and density estimation.   
(submitted by [Alex Yakubovich](https://github.com/ayakubovich))

<br>
<br>

<a class="mk-toclify" id="whitening-transformation"></a>
#### Whitening transformation

[[back to top](#table-of-contents)]

Whitening transformation is a normalization procedure to de-correlate samples in a dataset if the covariance matrix is not a diagonal matrix. Features are uncorrelated after "whitening" and their variances are equal unity, thus the covariance matrix becomes an identity matrix.

