




![logo](./Images/logo.png)

<hr>
**A collection of tutorials and examples for solving and understanding machine learning and pattern classification tasks.**
<br>
<br>





<br>
<br>

# Machine Learning and Pattern Classification

<br>
<br>

- [Machine learning and pattern classification with scikit-learn](#machine-learning-and-pattern-classification-with-scikit-learn)
- [Pre-Processing](#pre-processing)
- [Techniques for Dimensionality Reduction](#techniques-for-dimensionality-reduction)
- [Techniques for Parameter Estimation](#techniques-for-parameter-estimation)
- [Statistical Pattern Recognition Examples](#statistical-pattern-recognition-examples)
- [Links to useful resources](#links-to-useful-resources)






<br>
<br>
<br>
<hr>
<br>

### Machine learning and pattern classification with scikit-learn 
[[back to top](#machine-learning-and-pattern-classification)]

- Entry Point: Data - Using Python's sci-packages to prepare data for Machine Learning tasks and other data analyses [[IPython nb](http://nbviewer.ipython.org/github/rasbt/python_reference/blob/master/tutorials/python_data_entry_point.ipynb)]


- An Introduction to simple linear supervised classification using `scikit-learn` [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/machine_learning/scikit-learn/scikit_linear_classification.ipynb)]

<br>
<br>
<br>
<hr>
<br>

### Pre-processing

[[back to top](#machine-learning-and-pattern-classification)]

- About Feature Scaling: Standardization and Min-Max-Scaling (Normalization) [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/preprocessing/about_standardization_normalization.ipynb)]



<br>
<br>
<br>
<hr>
<br>

<hr>
<br>

### Techniques for Dimensionality Reduction
[[back to top](#machine-learning-and-pattern-classification)]

- **Projection**
	- Component Analyses
		- Linear Transformation
			- Principal Component Analysis (PCA) [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/dimensionality_reduction/projection/principal_component_analysis.ipynb)]
			- Linear Discriminant Analysis (LDA) [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/dimensionality_reduction/projection/linear_discriminant_analysis.ipynb)]



- **Feature Selection**
	- Sequential Feature Selection Algorithms [[IPython nb](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/dimensionality_reduction/feature_selection/sequential_selection_algorithms.ipynb)]


<br>
<hr>
<br>

### Techniques for Parameter Estimation
[[back to top](#machine-learning-and-pattern-classification)]


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


### Statistical Pattern Recognition Examples
[[back to top](#machine-learning-and-pattern-classification)]

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

- **Unsupervised Learning**


<br>
<hr>
<br>

## Links to useful resources
[[back to top](#machine-learning-and-pattern-classification)]

<br>
<br>


#### Dataset Collections
[[back to top](#machine-learning-and-pattern-classification)]

- [Kaggle](https://www.kaggle.com/competitions) - Kaggle, the leading platform for predictive modeling competitions. 

- [UCI MLR](http://archive.ics.uci.edu/ml/) - UC Irvine Machine Learning Repository

- [google.com/publicdata](http://www.google.com/publicdata/directory) - public data maintained by Google

- [Freebase](http://www.freebase.com) - A community-curated database of well-known people, places, and things

- [mldata.org](http://mldata.org) - machine learning data set repository for uploading and finding data sets

- [Infochimps](http://www.infochimps.com/datasets) - a huge collection of large-sized data sets


<br>
<br>

#### Specialized Datasets
[[back to top](#machine-learning-and-pattern-classification)]

- [Titanic Survivors](http://lib.stat.cmu.edu/S/Harrell/data/descriptions/titanic.html) - dataset with 1313 samples and 10 features about Titanic survivors

- [SMS Spam Collection](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/) - A collection of 425 SMS spam messages was manually extracted from the Grumbletext Web site

- [SNAP](http://snap.stanford.edu/data/index.html) - Stanford Large Network Dataset Collection

- [Amazon Google Books Ngrams](http://aws.amazon.com/datasets/8172056142375670) - A data set containing Google Books n-gram corpuses

- [The Million Song Dataset](http://labrosa.ee.columbia.edu/millionsong/) - Audio features and metadata for a million contemporary popular music tracks.

- [Modeling Online Auctions](http://www.modelingonlineauctions.com/datasets) - Datasets of bidding for different ebay auctions

- [CAT Dataset](http://137.189.35.203/WebUI/CatDatabase/catData.html) - A dataset of 10,000 cat images

- [Click Dataset](http://cnets.indiana.edu/groups/nan/webtraffic/click-dataset/) - A large dataset of about 53.5 billion HTTP requests made by users at Indiana University

- [Meteorites](http://www.analyticbridge.com/profiles/blogs/registered-meteorites-that-has-impacted-on-earth-visualized) - Registered meteorites that have impacted on Earth

- [Common Crawl 2012 web corpus](http://www.bigdatanews.com/profiles/blogs/big-data-set-3-5-billion-web-pages-made-available-for-all-of-us) - A hyperlink graph of 3.5 billion web pages and 128 billion hyperlinks between these pages

- [PyPi/Maven Dependency Data](http://ogirardot.wordpress.com/2013/01/31/sharing-pypimaven-dependency-data/) - State of the Maven/Java dependency graph and state of the PyPi/Python dependency graph.

- [NYPD Crash Data Band-Aid](http://nypd.openscrape.com/#/) - NYPD traffic crash data as a geocoded CSV

- [Pass rates, race & gender](http://home.cc.gatech.edu/ice-gt/556) - Detailed data on pass rates, race, and gender for 2013

- [Nominate/vote data](http://voteview.com/dwnl.htm) - Datasets including all the D-NOMINATE and W-NOMINATE scores

- [aiHit Datasets](http://endb-consolidated.aihit.com/datasets.htm) - Information on random 10,000 UK companies sampled from aiHit DB

- [Amsterdam Library of Object Images (ALOI)](http://aloi.science.uva.nl) - A color image collection of one-thousand small objects, recorded for scientific purposes
