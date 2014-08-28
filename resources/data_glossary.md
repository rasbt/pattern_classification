Sebastian Raschka  
last updated: 09/23/2014

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

- [Accuracy](#accuracy)
- [Aggregation](#aggregation)
- [Anomaly Detection](#anomaly-detection)
- [Backtesting](#backtesting)
- [Bagging](#bagging)
- [Batch Gradient Descent](#batch-gradient-descent)
- [Big Data](#big-data)
- [Bootstrapping](#bootstrapping)
- [Bregman divergence](#bregman-divergence)
- [Central Limit Theorem](#central-limit-theorem)
- [Confusion Matrix](#confusion-matrix)
- [Contingency Table](#contingency-table)
- [Correlation analysis](#correlation-analysis)
- [Correlation analysis - Kendall](##correlation-analysis---kendall)
- [Correlation analysis - Pearson](#correlation-analysis---pearson)
- [Correlation analysis - Spearman](#correlation-analysis---spearman)
- [Cosine Similarity](#cosine-similarity)
- [Cost function](#cost-function)
- [Covariate creation](#covariate-creation)
- [Cross-validation](#cross-validation)
- [Cross-validation, K-fold](#cross-validation-k-fold)
- [Cross-validation, Leave-One-Out](#cross-validation-leave-one-out)
- [Cross-validation, Random Sampling](#cross-validation-random-sampling)
- [Curse of dimensionality](#curse-of-dimensionality)
- [Data mining](#data-mining)
- [Decision rule](#decision-rule)
- [Decision tree classifier](#decision-tree-classifier)
- [Density-based clustering](#density-based-clustering)
- [Dimensionality reduction](#dimensionality-reduction)
- [Distance Metric Learning](#distance-metric-learning)
- [Distance, Euclidean](#distance-euclidean)
- [Distance, Manhattan](#distance-manhattan)
- [Distance, Minkowski](#distance-minkowski)
- [Eigenvectors and Eigenvalues](#eigenvectors-and-eigenvalues)
- [Ensemble methods](#ensemble-methods)
- [Evolutionary algorithms](#evolutionary-algorithms)
- [Exhaustive search](#exhaustive-search)
- [Feature Selection](#feature-selection )
- [Feature Space](#feature-space)
- [Fuzzy C-Means Clustering](#fuzzy-cmeans-clustering)
- [Generalization error](#generalization-error)
- [Genetic algorithm](#genetic-algorithm)
- [Gradient Descent](#gradient-descent)
- [Grid Search](#grid-search)
- [Heuristic search](#heuristic-search)
- [Hyperparameters](#hyperparameters)
- [Imputation](#imputation)
- [Independent Component Analysis](#independent-component-analysis)
- [Jaccard coefficient](#jaccard-coefficient)
- [Jackknifing](#jackknifing)
- [Kernel Density Estimation](#kernel-density-estimation)
- [Kernel Methods](#kernel-methods)
- [K-fold cross-validation](#k-fold-cross-validation)
- [K-Means Clustering](#k-means-clustering)
- [K-Means++ Clustering](#k-means-clustering-1)
- [K-Medoids Clustering](#k-medoids-clustering)
- [K-nearest neighbors algorithms](#k-nearest-neighbors-algorithms)
- [Knowledge Discovery in Databases (KDD)](#knowledge-discovery-in-databases-kdd)
- [LASSO Regression](#lasso-regression)
- [Law of Large Numbers](#law-of-large-numbers)
- [Least Squares fit](#least-squares-fit)
- [Least Squares fit - linear](#least-squares-fit-linear)
- [Linear Discriminant Analysis (LDA)](#linear-discriminant-analysis-lda)
- [Local Outlier Factor (LOF)](#local-outlier-factor-lof)
- [Logistic Regression](#logistic-regression)
- [Machine learning](#machine-learning)
- [Mahalanobis distance](#mahalanobis-distance)
- [MapRedcue](#mapreduce)
- [Markov chains](#markov-chains)
- [Maximum Likelihood Estimates (MLE)](#maximum-likelihood-estimates-mle)
- [Monte Carlo simulation](#monte-carlo-simulation)
- [Naive Bayes Classifier](#naive-bayes-classifier)
- [Normalization - Min-Max Scaling](#normalization-min-max-scaling)
- [Normalization - Standard Scores](#normalization-standard-scores)
- [Objective function](#objective-function)
- [Overfitting](#overfitting)
- [Parzen-Rosenblatt Window technique](#parzen-rosenblatt-window-technique)
- [Pattern classification](#pattern-classification)
- [Permissive transformations](#permissive-transformations)
- [Power transform](#power-transform)
- [Precision and Recall](#precision-and-recall)
- [Predictive Modeling](#predictive-modeling)
- [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
- [Proportion of Variance Explained (PVE)](#proportion-of-variance-explained-pve)
- [Purity Measure](#purity-measure)
- [Quantitative and qualitative attributes](#quantitative-and-qualitative-attributes)
- [Random forest](#random-forest)
- [Receiver Operating Characteristic (ROC)](#receiver-operating-characteristic-roc))
- [Regularization](#regularization)
- [Reinforcement learning](#reinforcement-learning)
- [Resubstitution error](#resubstitution-error)
- [Ridge Regression](#ridge-regression)
- [Sampling](#sampling)
- [Sensitivity](#sensitivity)
- [Specificity](#specificity)
- [Silhouette Measure (clustering)](#silhouette-measure-clustering)
- [Simple Matching Coefficient](#simple-matching-coefficient)
- [Singular Value Decomposition (SVD)](#singular-value-decomposition-svd)
- [Stochastic Gradient Descent (SGD)](#stochastic-gradient-descent-sgd)
- [Supervised learning](#supervised-learning)
- [Support Vector Machine (SVM)](#support-vector-machine)
- [Unsupervised learning](#unsupervised-learning)
- [White noise](#white-noise)
- [Whitening transformation](#whitening-transformation)


<br>
<br>

<a class="mk-toclify" id="accuracy"></a>
#### Accuracy
[[back to top](#table-of-contents)]

Accuracy is defined as the fraction of correct classifications out of the total number of samples; it resembles one way to assess the performance of a predictor and is often used synonymous to [specificity](#specificity)/[precision](#precision-and-recall) although it is calculated differently. Accuracy is calculated as (TP+TN)/(P+N), where TP=True Positives, TN=True Negatives, P=Positives, N=Negatives.

<br>
<br>


<a class="mk-toclify" id="aggregation"></a>
#### Aggregation
[[back to top](#table-of-contents)]

Aggregation is a data pre-processing procedure that is used to reduce the overall size of a dataset by combining attributes. An example of aggregation would be the averaging of daily temperature measurements to monthly temperature measurements.

<br>
<br>


<a class="mk-toclify" id="anomaly-detection"></a>
#### Anomaly detection
[[back to top](#table-of-contents)]

Anomaly detection describes the task of identifying points that deviate from specific patterns in a dataset --  the so-called outliers. Different types of anomaly detection methods include graph-based, statistical-based and distance-based techniques and can be used in both unsupervised and supervised learning tasks.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))
<br>
<br>

<a class="mk-toclify" id="backtesting"></a>
#### Backtesting
[[back to top](#table-of-contents)]

Backtesting is a specific case of [cross-validation](#cross-validation) in the context of finance and trading models where empirical data from previous time periods (data from the past) is used to evaluate a trading strategy.

<br>
<br>

<a class="mk-toclify" id="bagging"></a>
#### Bagging
[[back to top](#table-of-contents)]

Bagging is an ensemble method for classification (or regression analysis) in which individual models are trained by random sampling of data, and the final decision is made by voting among individual models with equal weights (or averaging for regression analysis).   
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>

<a class="mk-toclify" id="batch-gradient-descent-sgd"></a>
#### Batch Gradient Descent

[[back to top](#table-of-contents)]

Batch Gradient descent is a variant of a [Gradient Descent](#gradient-descent) algorithm to optimize a function by finding its local minimum. In contrast to [Stochastic Gradient Descent](#stochastic-gradient-descent-sgd) the gradient is computed from the whole dataset.

<br>
<br>

<a class="mk-toclify" id="big-data"></a>
#### Big Data
[[back to top](#table-of-contents)]

There are many different, controversial interpretations and definitions for the term "Big Data". Typically, one refers to data as "Big Data" if its volume and complexity are of a magnitude that the data cannot be processed by "conventional" computing platforms anymore; storage space, processing power, and database structures are typically among the limiting factors.

<br>
<br>

<a class="mk-toclify" id="bootstrapping"></a>
#### Bootstrapping
[[back to top](#table-of-contents)]

A resampling technique to that is closely related to [cross-validation](#cross-validation) where a training dataset is divided into random subsets. Bootstrapping -- in contrast to cross-validation -- is a random sampling **with** replacement. Bootstrapping is typically used for statistical estimation of bias and standard error, and a common application in machine learning is to estimate the generalization error of a predictor.

<br>
<br>

<a class="mk-toclify" id="bregman-divergence"></a>
#### Bregman divergence
[[back to top](#table-of-contents)]

Bregman divergence describes are family of proximity functions (or distance measures) that share common properties and are often used in clustering algorithms. A popular example is the squared Euclidean distance.

<br>
<br>

<a class="mk-toclify" id="central-limit-theorem"></a>
#### Central Limit Theorem
[[back to top](#table-of-contents)]

The Central Limit Theorem is a theorem in the field of probability theory that expresses the idea
that the distribution of sample means (from independent random variables) converges to a normal distribution when the sample size approaches infinity.
<br>
<br>

<a class="mk-toclify" id="confusion-matrix"></a>
#### Confusion Matrix
[[back to top](#table-of-contents)]

The confusion matrix is used as a way to represent the performance of a classifier and is sometimes also called "error matrix". This square matrix consists of columns and rows that list the number of instances as absolute or relative "actual class" vs. "predicted class" ratios.

<br>
<br>

<a class="mk-toclify" id="contingency-table"></a>
#### Contingency Table
[[back to top](#table-of-contents)]

A contingency table is used in clustering analysis to compare the overlap between two different clustering (grouping) results. The partitions from the two clustering results are represented as rows and columns in the table, and the individual elements of the table represent the number of elements that are shared between two partitions from each clustering result.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>

<a class="mk-toclify" id="correlation"></a>
#### Correlation analysis
[[back to top](#table-of-contents)]

Correlation analysis describes and quantifies the relationship between two independent variables. Typically, in case of a positive correlation both variables have a tendency to increase, and in the case of negative correlation, one variable increases while the other variable increases. It is important to mention the famous quotation "correlation does not imply causation".

<br>
<br>

<a class="mk-toclify" id="correlation-analysis---kendall"></a>
#### Correlation analysis - Kendall
[[back to top](#table-of-contents)]

Similar to the [Pearson correlation coefficient](#correlation-analysis-pearson), Kendall's tau measures the degree of a monotone relationship between variables, and like [Spearman's rho](#correlation-analysis-spearman), it calculates the dependence between ranked variables, which makes it feasible for non-normal distributed data. Kendall tau can be calculated for continuous as well as ordinal data. Roughly speaking, Kendall's tau distinguishes itself from Spearman's rho by stronger penalization of non-sequential (in context of the ranked variables) dislocations.

<br>
<br>

<a class="mk-toclify" id="correlation-analysis---pearson"></a>
#### Correlation analysis - Pearson
[[back to top](#table-of-contents)]

The Pearson correlation coefficient is probably the most widely used measure for linear relationships between two normal distributed variables and thus often just called "correlation coefficient". Usually, the Pearson coefficient is obtained via a [Least-Squares fit](#least-squares-fit) and a value of 1 represents a perfect positive relation-ship, -1 a perfect negative relationship, and 0 indicates the absence of a relationship between variables.

<br>
<br>

<a class="mk-toclify" id="correlation-analysis---spearman"></a>
#### Correlation analysis - Spearman
[[back to top](#table-of-contents)]

Related to the [Pearson correlation coefficient](#correlation-analysis-pearson), the Spearman correlation coefficient (rho) measures the relationship between two variables. Spearman's rho can be understood as a rank-based version of [Pearson's correlation coefficient](#correlation-analysis-pearson), which can be used for variables that are not normal-distributed and have a non-linear relationship. Also, its use is not only restricted to continuous data, but can also be used in analyses of ordinal attributes.

<br>
<br>

<a class="mk-toclify" id="cosine-similarity"></a>
#### Cosine Similarity
[[back to top](#table-of-contents)]

Cosine similarity measures the orientation of two *n*-dimensional sample vectors irrespective to their magnitude. It is calculated by the dot product of two numeric vectors, and it is normalized by the product of the vector lengths, so that output values close to 1 indicate high similarity.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>

<a class="mk-toclify" id="cosine-similarity"></a>
#### Cosine Similarity
[[back to top](#table-of-contents)]

Cosine similarity measures the orientation of two *n*-dimensional sample vectors irrespective to their magnitude. It is calculated by the dot product of two numeric vectors, and it is normalized by the product of the vector lengths, so that output values close to 1 indicate high similarity.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>

<a class="mk-toclify" id="cost-function"></a>
#### Cost function
[[back to top](#table-of-contents)]

A cost function (synonymous to loss function) is a special case of an [objective function](#objective-function), i.e., a function that is used for solving optimization problems. A cost function can take one or more input variables and the output variable is to be minimized. A typical use case for cost functions is parameter optimization.

<br>
<br>

<a class="mk-toclify" id="cross-validation"></a>
#### Cross-validation
[[back to top](#table-of-contents)]

Cross-validation is a statistical technique to estimate the prediction error rate by splitting the data into training, cross-validation, and test datasets. A prediction model is obtained using the training set, and model parameters are optimized by the cross-validation set, while the test set is held primarily for empirical error estimation.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>

<a class="mk-toclify" id="cross-validation-k-fold"></a>
#### Cross-validation, K-fold
[[back to top](#table-of-contents)]

K-fold cross-validation  is a variant of [cross validation](#cross-validation) where contiguous segments of samples are selected from the training dataset to build two new subsets for every iteration (without replacement): a new training and test dataset (while the original test dataset is retained for the final evaluation of the predictor). 

<br>
<br>

<a class="mk-toclify" id="cross-validation-leave-one-out"></a>
#### Cross-validation, Leave-One-Out
[[back to top](#table-of-contents)]

Leave-One-Out cross-validation a variant of [cross validation](#cross-validation) one sample is removed for every iteration (without replacement). The model is trained on the remaining N-1 samples and evaluated via the removed sample (while the original test dataset is retained for the final evaluation of the predictor). 

<br>
<br>

<a class="mk-toclify" id="cross-validation-random-sampling"></a>
#### Cross-validation, Random Sampling
[[back to top](#table-of-contents)]

Cross-validation via random sampling is a variant of [cross validation](#cross-validation) where random chunks of samples are extracted from the training dataset to build two new subsets for every iteration  (with or without replacement): a new training and test dataset for every iteration (while the original test dataset is retained for the final evaluation of the predictor). 



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

A field that is closely related to machine learning and pattern classification. The focus of data mining does not lie in merely the collection of data, but the extraction of useful information: Discovery of patterns, and making inferences and predictions. Common techniques in data mining include predictive modeling, classification, and anomaly detection.

<br>
<br>

<a class="mk-toclify" id="dbscan"></a>
#### DBSCAN
[[back to top](#table-of-contents)]

DBSCAN is a variant of a density-based clustering algorithm that identifies core points as regions of high-densities based on their number of neighbors (> *MinPts*) in a specified radius (&epsilon;). Points that are below *MinPts* but within &epsilon; are specified as border points; the remaining points are classified as noise points.   
(submitted by [Vahid Mirjalili](https://github.com/mirjalil); edited)

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
#### Dimensionality reduction
[[back to top](#table-of-contents)]

Dimensionality reduction is a data pre-processing step in machine learning applications that aims to avoid the [curse of dimensionality](#curse-of-dimensionality) and reduce the effect of overfitting.  Dimensionality reduction is related to [feature selection](#feature-selection), but instead of selecting a feature subset, dimensionality reduction takes as projection-based approach (e.g, linear transformation) in order to create a new feature subspace.

<br>
<br>

<a class="mc-toclify" id="distance-metric-learning"></a>
#### Distance Metric Learning
[[back to top](#table-of-contents)]

Distance metrics are fundamental for many machine learning algorithms. Distance metric learning - instead of learning a model - incorporates estimated relevances of features to obtain a distance metric for potentially optimal separation of classes and clusters: Large distances for objects from different classes, and small distances for objects of the same class, respectively.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil); edited)

<br>
<br>

<a class="mc-toclify" id="distance-euclidean"></a>
#### Distance, Euclidean
[[back to top](#table-of-contents)]

The Euclidean distance is a distance measure between two points or or vectors in a two- or multidimensional (Euclidean) space based on Pythagoras' theorem. The distance is calculated by taking the square root of the sum of the squared pair-wise distances of every dimension.  

![](../Images/latex_equations/distance_euclidean_1.gif)

<br>
<br>

<a class="mc-toclify" id="distance-manhattan"></a>
#### Distance, Manhattan
[[back to top](#table-of-contents)]

The Manhattan distance (sometimes also called Taxicab distance) metric is related to the Euclidean distance, but instead of calculating the shortest diagonal path ("beeline") between two points, it calculates the distance based on gridlines. The Manhattan distance was named after the block-like layout of the streets in Manhattan. 

![](../Images/latex_equations/distance_manhattan_1.gif)


<br>
<br>


<a class="mc-toclify" id="distance-minkowski"></a>
#### Distance, Minkowski
[[back to top](#table-of-contents)]

The Minkowski distance is a generalized form of the Euclidean distance (if *p=2*) and the Manhattan distance (if *p=1*).  

![](../Images/latex_equations/distance_minkowski_1.gif)

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

<a class="mc-toclify" id="ensemble-methods"></a>
#### Ensemble methods
[[back to top](#table-of-contents)]

Ensemble methods combine multiple classifiers which may differ in algorithms, input features, or input samples. Statistical analyses showed that ensemble methods yield better classification performances and are also less prone to overfitting. Different methods, e.g., bagging or boosting, are used to construct the final classification decision based on weighted votes.  

<br>
<br>

<a class="mc-toclify" id="evolutionary-algorithms"></a>
#### Evolutionary algorithms
[[back to top](#table-of-contents)]

Evolutionary algorithms are a class of algorithms that are based on [heuristic search](#heuristic-search) methods inspired by biological evolution in order to solve optimization problems.

<br>
<br>

<a class="mk-toclify" id="exhaustive-search"></a>
#### Exhaustive search
[[back to top](#table-of-contents)]

Exhaustive search (synonymous to brute-force search) is a problem-solving approach where all possible combinations are sequentially evaluated to find the optimal solution. Exhaustive search guarantees to find the optimal solution whereas other approaches (e.g., [heuristic searches](#heuristic-search)) are regarded as sub-optimal. A downside of exhaustive searches is that computational costs increase proportional to the number of combinations to be evaluated.


<br>
<br>

<a class="mk-toclify" id="feature-selection"></a>
#### Feature Selection
[[back to top](#table-of-contents)]

Feature selection is an important pre-processing step in many machine learning applications in order to avoid the [curse of dimensionality](#curse-of-dimensionality) and [overfitting](#overfitting). A subset of features is typically selected by evaluating different combinations of features and eventually retain the subset that minimizes a specified [cost function](#cost-function). Commonly used algorithms for feature selection as alternative to [exhaustive search](#exhaustive-search) algorithms include sequential selection algorithms and genetic algorithms

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

<a class="mk-toclify" id="generalization-error"></a>
#### Generalization error
[[back to top](#table-of-contents)]

The generalization error describes how well new data can be classified and is a useful metric to assess the performance of a classifier. Typically, the generalization error is computed via [cross-validation](cross-validation) or simply the absolute difference between the error rate on the training and test dataset.

<br>
<br>

<a class="mc-toclify" id="evolutionary-algorithms"></a>
#### Genetic algorithm
[[back to top](#table-of-contents)]

The Genetic algorithm is a subclass of [evolutionary algorithms](#evolutionary-algorithms) that takes a heuristic approach inspired by Charles Darwin's theory of "natural selection" in order to solve optimization problems.

<br>
<br>

<a class="mk-toclify" id="gradient-descent"></a>
#### Gradient Descent
[[back to top](#table-of-contents)]

Gradient descent is an algorithm that optimizes a function by finding its local minimum. After the algorithm was initialized with an initial guess, it takes the derivative of the function to make a step towards the direction of deepest descent. This step-wise process is repeated until convergence.

<br>
<br>

<a class="mk-toclify" id="grid-search"></a>
#### Grid Search
[[back to top](#table-of-contents)]

Grid Search is a procedure to optimize parameters of a learning algorithm that are not learned from the training data (i.e., [hyperparameters](#hyperparameters)). Typically, Grid Search is implemented as an exhaustive search (in contrast to randomized parameter optimization) of candidate parameter values. After all possible parameter combination for a model are evaluated, the best combination will be retained.

<br>
<br>

<a class="mk-toclify" id="heuristic-search"></a>
#### Heuristic search
[[back to top](#table-of-contents)]

Heuristic search is a problem-solving approach that is focussed on efficiency rather than completeness in order to find a suboptimal solution to a problem. Heuristic search is often used as alternative approach where [exhaustive search](#exhaustive-search) is too computationally intensive and where solutions need to be approximated.


<br>
<br>

<a class="mk-toclify" id="hyperparameters"></a>
#### Hyperparameters
[[back to top](#table-of-contents)]

Hyperparameters are the parameters of a classifier or estimator that are not directly learned in the machine learning step from the training data but are optimized separately (e.g., via [Grid Search](#grid-search)). The goal of hyperparameter optimization is to achieve good generalization of a learning algorithm and to avoid overfitting to the training data.

<br>
<br>

<a class="mk-toclify" id="imputation"></a>
#### Imputation
[[back to top](#table-of-contents)]

Imputations algorithms are designed to replace the missing data (NAs) with certain statistics rather than discarding them for downstream analysis. Commonly used imputation methods include mean imputation (replacement by the sample mean of an attribute), kNN imputation, and regression imputation.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil); edited)

<br>
<br>

<a class="mk-toclify" id="independent-component-analysis"></a>
#### Independent Component Analysis
[[back to top](#table-of-contents)]

Independent Component Analysis (ICA) is a statistical signal-processing technique that decomposes a multivariate dataset of mixed, non-gaussian distributed source signals into  independent components 
A popular example is the separation of overlapping voice samples -- the so-called "cocktail party problem".

<br>
<br>

<a class="mk-toclify" id="jaccard-coefficient"></a>
#### Jaccard coefficient
[[back to top](#table-of-contents)]

The Jaccard coefficient is used as similarity measure for asymmetric binary data and calculated by taking the number of matching attributes and divide it by the number of all attributes except those where both variables have a value 0 in contrast to a [simple matching coefficient](#simple-matching -coefficient).

<br>
<br>

<a class="mk-toclify" id="jackknifing"></a>
#### Jackknifing
[[back to top](#table-of-contents)]

Jackknifing is a resampling technique that predates the related [cross-validation](#cross-validation) and [bootstrapping](#bootstrapping) techniques and is mostly used for bias and variance estimations. In jackknifing, a dataset is split into N subsets where exactly one sample is removed from every subset so that every subset is of size N-1.

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

<a class="mk-toclify" id="k-nearest-neighbors-algorithms"></a>
#### K-nearest neighbors algorithms  
[[back to top](#table-of-contents)]  

K-nearest neighbors algorithms find the k-points that are closest to a point of interest based on their attributes using a certain distance measure (e.g., Euclidean distance). K-nearest neighbors algorithms are being used in many different contexts: Non-parametric density estimation, missing value imputation, dimensionality reduction, and classifiers in supervised and unsupervised pattern classification and regression problems.

<br>
<br>

<a class="mk-toclify" id="knowledge-discovery-in-databases-kdd"></a>
#### Knowledge Discovery in Databases (KDD)  
[[back to top](#table-of-contents)]

Knowledge Discovery in Databases (KDD) describes a popular workflow in the field of data mining for extracting useful and meaningful information. Typically, the individual steps consist of data preprocessing (feature selection, dimensionality reduction, normalization, etc.), the discovery of patterns, post-processing, and eventually information retrieval.


<br>
<br>

<a class="mk-toclify" id="lasso-regression"></a>
#### LASSO Regression
[[back to top](#table-of-contents)]

LASSO (Least Absolute Shrinkage and Selection Operator) is a regression model that uses the L1-norm (sum of absolute values) of model coefficients to penalize the model complexity. LASSO has the advantage that some coefficients can become zero, as opposed to [ridge regression](#ridge-regression) that uses the squared sum of model coefficients.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>

<a class="mk-toclify" id="law-of-large-numbers"></a>
#### Law of Large Numbers
[[back to top](#table-of-contents)]

The Law of Large Numbers is a theorem in the field of probability theory that expresses the idea that the actual value of a random sampling process approaches the expected value for growing sample sizes. A common example is that the observed ratio of "heads" in an unbiased coin-flip experiment will approach 0.5 for large sample sizes. 

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

<a class="mk-toclify" id="local-outlier-factor"></a>
#### Local Outlier Factor (LOF)
[[back to top](#table-of-contents)]

LOF is a density-based [anomaly detection](#anomaly-detection) technique for outlier identification. The LOF for a point *p* refers to the average "reachability distance" towards its nearest neighbors. Eventually, the points with the largest LOF values (given a particular threshold) are identified as outliers.   
(submitted by [Vahid Mirjalili](https://github.com/mirjalil); edited)

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

<a class="mk-toclify" id="mapreduce"></a>
#### MapReduce

[[back to top](#table-of-contents)]

MapRedcue is a programming model for analyzing large datasets on distributed computer clusters, in which the task is divided into two steps, a map step and a reducer step. In the map step, the data are filtered by some factors on each compute node, then filtered data are shuffled and passed to the reducer function which performs further analysis on each portion of filtered data separately.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

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

A classifier based on a statistical model (i.e., Bayes theorem: calculating posterior probabilities based on the prior probability and the so-called likelihood) in the field of pattern classification. Naive Bayes assumes that all attributes are conditionally independent, thereby, computing the likelihood is simplified to the product of the conditional probabilities of observing individual attributes given a particular class label. 

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

<a class="mk-toclify" id="objective-function"></a>
#### Objective function
[[back to top](#table-of-contents)]

Objective functions are mathematical function that are used for problem-solving and optimization tasks. Depending on the task, the objective function can be omtpimized through minimization ([cost or loss functions](#cost-function)) or maximization (reward function). A typical application of an objective function in pattern classification tasks is to minimize the error rate of a classifier.

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


<a class="mk-toclify" id="permissive-transformations"></a>
#### Permissive transformations

[[back to top](#table-of-contents)]

Permissive transformations are transformations of data that that do not change the "meaning" of the attributes, such as scaling or mapping. For example, the transformation of temperature measurements from a Celsius to a Kelvin scale would be a permissive transformation of a numerical attribute.

<br>
<br>


<a class="mk-toclify" id="power-transform"></a>
#### Power transform

[[back to top](#table-of-contents)]

Power transforms form a category of statistical transformation techniques that are used to transform non-normal distributed data to normality.

<br>
<br>

<a class="mk-toclify" id="principal-component-analysis-pca"></a>
#### Principal Component Analysis (PCA)
[[back to top](#table-of-contents)]

A linear transformation technique that is commonly used to project a dataset (without utilizing class labels) onto a new feature space or feature subspace (for dimensionality reduction) where the new component axes are the directions that maximize the variance/spread of the data. 

<br>
<br>

<a class="mk-toclify" id="precision-and-recall"></a>
#### Precision and Recall
[[back to top](#table-of-contents)]

Precision (synonymous to [specificity](#specificity)) and recall (synonymous to [sensitivity](#sensitivity)) are two measures to assess performance of a classifier if class label distributions are skewed.
Precision is defined as the ratio of number of relevant items out of total retrieved items, whereas recall is the fraction of relevant items which are retrieved.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil); edited)

<br>
<br>

<a class="mk-toclify" id="predictive-modeling"></a>
#### Predictive Modeling
[[back to top](#table-of-contents)]

Predictive modeling is a [data mining](#data-mining) technique that describes the process of collecting data and training a classifier. Eventually, the the goal is to create a statistical model that can be used to make predictions about future trends and behaviors, e.g., predictions of stock market trends or the classifications of spam vs. non-spam e-mails.

<br>
<br>

<a class="mk-toclify" id="proportion-of-variance-explained-pve"></a>
#### Proportion of Variance Explained
[[back to top](#table-of-contents)]

In the context of dimensionality reduction, the proportion of variance explained (PVE) describes how much of the total variance is captured by the new selected axes, for example,  principal components or discriminant axes. It is computed by the sum of variance of new component axes divided by the total variance.   
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>

<a class="mk-toclify" id="purity-measure"></a>
#### Purity Measure

[[back to top](#table-of-contents)]

In a cluster analysis with given truth cluster memberships (or classes), "purity" is used to assess the effectiveness of clustering. Purity is measured by assigning each cluster to the class that is maximally represented and computed via the weighted average of maximum number of samples from the same class in each cluster.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>

<a class="mk-toclify" id="quantitative-and-qualitative-attributes"></a>

#### Quantitative and qualitative attributes    
[[back to top](#table-of-contents)]

Quantitative attributes are also often called "numeric"; those are attributes for which calculations and comparisons like ratios and intervals make sense (e.g., temperature in Celsius). Qualitative, or "categorical", attributes can be grouped into to subclasses: nominal and ordinal. Where ordinal attributes (e.g., street numbers) can be ordered, nominal attributes can only distinguished by their category names (e.g., colors). 

<br>
<br>

<a class="mk-toclify" id="random-forest"></a>
#### Random forest

[[back to top](#table-of-contents)]

Random forest is an ensemble classifier where multiple [decision tree classifiers](#decision-tree-classifier) are learned and combined via the [bagging](#bagging) technique. Unseen/test objects are then classified by taking the majority of votes from individual decision trees.
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>

<a class="mk-toclify" id="receiver-operating-characteristic-roc"></a>
#### Receiver Operating Characteristic (ROC)

[[back to top](#table-of-contents)]

The Receiver Operating Characteristic (ROC, or ROC curve) is a quality measure for binary prediction algorithms by plotting the "False positive rate" vs. the "True positive rate" ([sensitivity](#sensitivity)).

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

<a class="mk-toclify" id="resubstitution-error"></a>

#### Resubstitution error
[[back to top](#table-of-contents)]

The resubstitution error represents the classification error rate on the training dataset (the dataset that was used to train the classifier). The performance of a classifier cannot be directly deduced from resubstitution error alone, but it becomes a useful measure for calculating the [generalization error](#generalization-error).

<br>
<br>



<a class="mk-toclify" id="ridge-regression"></a>
#### Ridge Regression

[[back to top](#table-of-contents)]

Ridge regression is a regularized regression technique in which the squared sum of the model coefficients is used to penalize model complexity.   
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>


<a class="mk-toclify" id="sampling"></a>
#### Sampling

[[back to top](#table-of-contents)]

Sampling is data pre-processing procedure that is used to reduce the overall size of a dataset and to reduce computational costs by selecting a representative subset from the whole input dataset.

<br>
<br>

<a class="mk-toclify" id="sensitivity"></a>
#### Sensitivity

[[back to top](#table-of-contents)]

Sensitivity (synonymous to [precision](#precision-and-recall)), which is related to [specificity](#specificity) -- in the context of error rate evaluation -- describes the "True Positive Rate" for a binary classification problem: The probability to make a correct prediction for a "positive/true" case (e.g., in an attempt to predict a disease, the disease is correctly predicted for a patient who truly has this disease). Sensitivity is calculated as (TP)/(TP+FN), where TP=True Positives, FN=False Negatives.

<br>
<br>



<a class="mk-toclify" id="silhouette-measure-clustering"></a>
#### Silhouette Measure (clustering)

[[back to top](#table-of-contents)]

Silhouette measure provides a metric to evaluate the performance of a clustering analysis. For each data point *i*, it measures the average distance of point *i* to all other points in the same cluster *(a(i))*, and the minimum distance to points from other clusters *(b(i))*. The average silhouette measures for each cluster can provide a visual way to pick the proper number of clusters.  
(submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>

<a class="mk-toclify" id="simple-matching-coefficient"></a>
#### Simple Matching Coefficient

[[back to top](#table-of-contents)]

The simple matching coefficient is a similarity measure for binary data and calculated by dividing the total number of matches by the total number of attributes. For asymmetric binary data, the related [Jaccard coefficient](#jaccard-coefficient) is to be preferred in order to avoid highly similar scores.

<br>
<br>

<a class="mk-toclify" id="singular-value-decomposition-svd"></a>
#### Singular Value Decomposition (SVD)

[[back to top](#table-of-contents)]

Singular value decomposition (SVD) is linear algebra technique that decomposes matrix ***X*** into  
***U D V<sup>T</sup>*** where ***U*** (left-singular vectors) and ***V*** (right-singular vector) are both column-orthogonal, and D is a diagonal matrix that contains singular values. [PCA](#principal-component-analysis) is closely related to the right0singular vectors of SVD.  
 (submitted by [Vahid Mirjalili](https://github.com/mirjalil))

<br>
<br>

<a class="mk-toclify" id="specificity"></a>
#### Specificity

[[back to top](#table-of-contents)]

Specificity (synonymous to [recall](#precision-and-recall)), which is related to [sensitivity](#sensitivity) -- in the context of error rate evaluation -- describes the "True Negative Rate" for a binary classification problem: The probability to make a correct prediction for a "false/negative" case (e.g., in an attempt to predict a disease, no disease is predicted for a healthy patient). Specificity is calculated as (TN)/(FP+TN), where TN=True Negatives, FP=False Positives.

<br>
<br>

<a class="mk-toclify" id="stochastic-gradient-descent-sgd"></a>
#### Stochastic Gradient Descent (SGD)

[[back to top](#table-of-contents)]

Stochastic Gradient Descent (SGD) (also see [Gradient Descent](#gradient-descent)) is a machine learning algorithm that seeks to minimize an objective (or cost) function and can be grouped into the category of linear classifiers for supervised learning tasks. In contrast to [Batch Gradient Descent](#batch-gradient-descent), the gradient is computed from a single sample.
<br>
<br>

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


<a class="mk-toclify" id="white-noise"></a>
#### White noise
[[back to top](#table-of-contents)]

White noise is a source that produces random, statistically independent variables following a particular distribution. In the field of sound processing, white noise is also often referred to as a mixture of tones or sounds of different frequencies.

<br>
<br>


<a class="mk-toclify" id="whitening-transformation"></a>
#### Whitening transformation

[[back to top](#table-of-contents)]

Whitening transformation is a normalization procedure to de-correlate samples in a dataset if the covariance matrix is not a diagonal matrix. Features are uncorrelated after "whitening" and their variances are equal unity, thus the covariance matrix becomes an identity matrix.



