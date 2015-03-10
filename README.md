




![logo](./Images/logo.png)

<hr>
**Tutorials, examples, collections, and everything else that falls into the categories: pattern classification, machine learning, and data mining.**
<br>
<br>



<br>
<br>

# Sections


- [Introduction to Machine Learning and Pattern Classification](#introduction-to-machine-learning-and-pattern-classification)
- [Pre-Processing](#pre-processing)
- [Model Evaluation](#model-evaluation)
- [Parameter Estimation](#parameter-estimation)
- [Machine Learning Algorithms and Classification Models](#machine-learning-algorithms-and-classification-models)
- [Clustering](#clustering)
- [Collecting Data](#collecting-data)
- [Data Visualization](#data-visualization)
- [Statistical Pattern Classification Examples](#statistical-pattern-classification-examples)
- [Talks](#talks)
- [Resources](#resources)

<br>



<br>

<img src="./Images/supervised_learning_flowchart.png" style="width: 700px; height:600px;">

[[Download a PDF version](https://github.com/rasbt/pattern_classification/raw/master/PDFs/supervised_learning_flowchart.pdf)] of this flowchart.

<br>
<br>
<br>
<hr>
<br>

### Introduction to Machine Learning and Pattern Classification
[[back to top](#sections)]

- Predictive modeling, supervised machine learning, and pattern classification - the big picture [[Markdown](./machine_learning/supervised_intro/introduction_to_supervised_machine_learning.md)]

- Entry Point: Data - Using Python's sci-packages to prepare data for Machine Learning tasks and other data analyses [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/machine_learning/scikit-learn/python_data_entry_point.ipynb)]

- An Introduction to simple linear supervised classification using `scikit-learn` [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/machine_learning/scikit-learn/scikit_linear_classification.ipynb)]



<br>
<br>
<br>
<hr>
<br>

### Pre-processing

[[back to top](#sections)]

- **Feature Extraction**
	- Tips and Tricks for Encoding Categorical Features in Classification Tasks [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/preprocessing/feature_encoding.ipynb)]
- **Scaling and Normalization**
	- About Feature Scaling: Standardization and Min-Max-Scaling (Normalization) [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/preprocessing/about_standardization_normalization.ipynb)]


- **Feature Selection**
	- Sequential Feature Selection Algorithms [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/dimensionality_reduction/feature_selection/sequential_selection_algorithms.ipynb)] 

- **Dimensionality Reduction**
	- Principal Component Analysis (PCA) [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/dimensionality_reduction/projection/principal_component_analysis.ipynb)] 
	- The effect of scaling and mean centering of variables prior to a PCA [[PDF](https://github.com/rasbt/pattern_classification/raw/master/dimensionality_reduction/projection/scale_center_pca/scale_center_pca.pdf)] [[HTML](http://htmlpreview.github.io/?https://raw.githubusercontent.com/rasbt/pattern_classification/master/dimensionality_reduction/projection/scale_center_pca/scale_center_pca.html)] 
	- PCA based on the covariance vs. correlation matrix  [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/dimensionality_reduction/projection/pca_cov_cor.ipynb)] 

	- Linear Discriminant Analysis (LDA) [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/dimensionality_reduction/projection/linear_discriminant_analysis.ipynb)]
	- Kernel tricks and nonlinear dimensionality reduction via PCA [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/dimensionality_reduction/projection/kernel_pca.ipynb)]

<br>
<hr>
<br>

### Model Evaluation
[[back to top](#sections)]

- An Overview of General Performance Metrics of Binary Classifier Systems [[PDF](http://sebastianraschka.com/PDFs/articles/performance_metrics.pdf)]
- **Cross-validation**
	- Streamline your cross-validation workflow - scikit-learn's Pipeline in action [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/machine_learning/scikit-learn/scikit-pipeline.ipynb)]

<br>
<hr>
<br>

### Parameter Estimation
[[back to top](#sections)]

- **Parametric Techniques**
    - Introduction to the Maximum Likelihood Estimate (MLE) [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/parameter_estimation_techniques/maximum_likelihood_estimate.ipynb)]
    - How to calculate Maximum Likelihood Estimates (MLE) for different distributions [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/parameter_estimation_techniques/max_likelihood_est_distributions.ipynb)]
	
- **Non-Parametric Techniques**
	- Kernel density estimation via the Parzen-window technique [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/parameter_estimation_techniques/parzen_window_technique.ipynb)]
	- The K-Nearest Neighbor (KNN) technique 


- **Regression Analysis**
	- Linear Regression
		- Least-Squares fit [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/data_fitting/regression/linregr_least_squares_fit.ipynb)]
   
   - Non-Linear Regression

<br>
<hr>
<br>


### Machine Learning Algorithms and Classification Models
[[back to top](#sections)]

- Naive Bayes and Text Classification I - Introduction and Theory [[View PDF](http://sebastianraschka.com/PDFs/articles/naive_bayes_1.pdf)] [[Download PDF](https://github.com/rasbt/pattern_classification/raw/master/machine_learning/naive_bayes_1/tex/naive_bayes_1.pdf)] 

- Implementing a Weighted Majority Rule Ensemble Classifier in scikit-learn  [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/machine_learning/scikit-learn/ensemble_classifier.ipynb)]

<br>
<hr>
<br>

### Clustering
[[back to top](#sections)]

- **Protoype-based clustering**
- **Hierarchical clustering**
	- Complete-Linkage Clustering and Heatmaps in Python [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/clustering/hierarchical/clust_complete_linkage.ipynb)]
- **Density-based clustering**
- **Graph-based clustering**
- **Probabilistic-based clustering**

<br>
<hr>
<br>



## Collecting Data
[[back to top](#sections)]

- Collecting Fantasy Soccer Data with Python and Beautiful Soup [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/data_collecting/parse_dreamteamfc_data.ipynb)]

- Download Your Twitter Timeline and Turn into a Word Cloud Using Python [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/data_collecting/twitter_wordcloud.ipynb)]

<br>
<hr>
<br>

## Data Visualization
[[back to top](#sections)]

- Exploratory Analysis of the Star Wars API  [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/data_viz/swapi_viz.ipynb)]

![](./Images/data_viz/data_viz_swapi_1.png)

- Matplotlib examples -Exploratory data analysis of the Iris dataset [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/data_viz/matplotlib_viz_gallery.ipynb)]

![](./Images/data_viz/data_viz_iris_1.png)
<br>
<hr>
<br>


### Statistical Pattern Classification Examples
[[back to top](#sections)]

- **Supervised Learning**
    	
    - Parametric Techniques
    	- Univariate Normal Density 
    		- Ex1: 2-classes, equal variances, equal priors [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/stat_pattern_class/supervised/parametric/1_stat_superv_parametric.ipynb)]
			- Ex2: 2-classes, different variances, equal priors [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/stat_pattern_class/supervised/parametric/2_stat_superv_parametric.ipynb)]
			- Ex3: 2-classes, equal variances, different priors [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/stat_pattern_class/supervised/parametric/3_stat_superv_parametric.ipynb)]
			- Ex4: 2-classes, different variances, different priors, loss function [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/stat_pattern_class/supervised/parametric/4_stat_superv_parametric.ipynb)]
			- Ex5: 2-classes, different variances, equal priors, loss function, cauchy distr. [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/stat_pattern_class/supervised/parametric/5_stat_superv_parametric.ipynb)]			
			
			
			
    	- Multivariate Normal Density
			- Ex5: 2-classes, different variances, equal priors, loss function [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/stat_pattern_class/supervised/parametric/5_stat_superv_parametric.ipynb)]
			- Ex7: 2-classes, equal variances, equal priors [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/stat_pattern_class/supervised/parametric/7_stat_superv_parametric.ipynb)]  		
    		
    - Non-Parametric Techniques


<br>
<hr>
<br>

## Talks
[[back to top](#sections)]


<iframe src="//www.slideshare.net/slideshow/embed_code/44568310" width="340" height="290" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC; border-width:1px; margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe> <div style="margin-bottom:5px"> <strong> <a href="//www.slideshare.net/SebastianRaschka/nextgen-talk-022015" title="An Introduction to Supervised Machine Learning and Pattern Classification: The Big Picture" target="_blank">An Introduction to Supervised Machine Learning and Pattern Classification: The Big Picture</a> </strong> from <strong><a href="//www.slideshare.net/SebastianRaschka" target="_blank">Sebastian Raschka</a></strong> </div>

<iframe src="//www.slideshare.net/slideshow/embed_code/42579528" width="340" height="290" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC; border-width:1px; margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe> <div style="margin-bottom:5px"> <strong> <a href="//www.slideshare.net/SebastianRaschka/musicmood-20140912" title="MusicMood - Machine Learning in Automatic Music Mood Prediction Based on Song Lyrics" target="_blank">MusicMood - Machine Learning in Automatic Music Mood Prediction Based on Song Lyrics</a> </strong> from <strong><a href="//www.slideshare.net/SebastianRaschka" target="_blank">Sebastian Raschka</a></strong> </div>

<br>
<hr>
<br>

## Resources
[[back to top](#sections)]



- Copy-and-paste ready LaTex equations [[Markdown](./resources/latex_equations.md)]

- Open-source datasets [[Markdown](./resources/dataset_collections.md)]

- Free Machine Learning eBooks [[Markdown](./resources/machine_learning_ebooks.md)]

- Terms in data science defined in less than 50 words [[Markdown](./resources/data_glossary.md)]

- Useful libraries for data science in Python [[Markdown](./resources/python_data_libraries.md)]

- General Tips and Advices [[Markdown](./resources/general_tips_and_advices.md)]

- A matrix cheatsheat for Python, R, Julia, and MATLAB  [[HTML](http://sebastianraschka.com/github/pattern_classification/matrix_cheatsheet_table.html)]





