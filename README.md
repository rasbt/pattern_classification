pattern_classification
======================

<hr>
**A collection of tutorials and examples for solving and understanding machine learning and pattern classification tasks.**
<br>
<br>

<font size=5em>
**Newest**: <a href="#parzen">Kernel density estimation via the Parzen-window technique</a></font>

<hr>
<br>
<a name="sections"></a>
<br>
<br>
# Sections
<br>
&nbsp;&#8226; <a href="#dim_red"><strong>Techniques for Dimensionality 
Reduction</strong></a><br>



&nbsp;&nbsp;&nbsp;&#8226; <a href="#projection">Projection</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; <a href="#comp_analysis">Component Analyses</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; <a href="#linear_transf">Linear Transformation</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; <a href="#pca">Principal Component Analysis (PCA)</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; <a href="#mda">Multiple Discriminant Analysis (MDA)</a><br>

&nbsp;&nbsp;&nbsp;&#8226; <a href="#feat_sel"><strong>Feature Selection</a></strong><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; <a href="#seq_feat_sel_algos">Sequential Feature Selection Algorithms</a><br>

&nbsp;&#8226; <a href="#est_param_tech"><strong>Techniques for Parameter Estimation</strong></a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; <a href="#est_param">Parametric Techniques</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; <a href="#mle">Introduction to the Maximum Likelihood Estimate (MLE)</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; <a href="#mle_dist">How to calculate Maximum Likelihood Estimates (MLE) for different distributions</a>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; <a href="#est_nonparam"> Non-Parametric Techniques</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; <a href="#parzen">Kernel density estimation via the Parzen-window technique</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; The K-Nearest Neighbor (KNN) technique
<br>
<br>
&#8226; <a href="#stat_pat_rec"><strong>Statistical Pattern Recognition Examples</strong></a><br>

&nbsp;&nbsp;&nbsp;&#8226; <a href="#supervised">Supervised Learning</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; <a href="#param">Parametric Techniques</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; <a href="#univar">Univariate Normal Density</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; <a href="#multivar">Multivariate Normal Density</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; Non-Parametric Techniques<br>


&nbsp;&nbsp;&nbsp;&#8226; Unsupervised Learning<br>

<hr>




<br>
<br>


<br>
<br>
<hr>
<p><a name="dim_red"></a></p>

#Techniques for Dimensionality Reduction
[[back to top](#sections)]


<hr>

<p><a name="projection"></a></p>

## Projection
[[back to top](#sections)]

<p><a name="comp_analyses"></a></p>

### Component Analyses
[[back to top](#sections)]

<p><a name="linear_transf"></a></p>

### Linear Transformation
[[back to top](#sections)]
<br>
<br>


<p><a name="pca"></a></p>

#### Principal Component Analyses (PCA)
[[back to top](#sections)]
<br><br>
![./Images/principal_component_analysis.png](./Images/principal_component_analysis.png)
<br><br>
[View IPython Notebook](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/dimensionality_reduction/projection/principal_component_analysis.ipynb?create=1)  

[Download PDF](https://github.com/rasbt/pattern_classification/raw/master/PDFs/principal_component_analysis_sebastian_raschka.pdf)

<p><a name="mda"></a></p>
<br>
<br>

#### Multiple Discriminant Analysis (MDA)
[[back to top](#sections)]
<br><br>
![./Images/mda_overview2.png](./Images/mda_overview2.png)
<br><br>
[View IPython Notebook](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/dimensionality_reduction/projection/multiple_discriminant_analysis.ipynb?create=1)  
<br>
<br>
<br>
<br>


<p><a name="feat_sel"></a></p>

## Feature Selection
[[back to top](#sections)]
<p><a name="seq_feat_sel_algos"></a></p>

#### Sequential Feature Selection Algorithms
[[back to top](#sections)]
![](./Images/feat_sele_alg.png)

[View IPython Notebook](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/dimensionality_reduction/feature_selection/sequential_selection_algorithms.ipynb?create=1)  
<br>
[Download PDF](https://github.com/rasbt/pattern_classification/tree/master/dimensionality_reduction/feature_selection/sequential_selection_algorithms.pdf)
<br>
<br>
<br>
<br>
<p><a name="est_param_tech"></a></p>
## Techniques for Parameter Estimation
[[back to top](#sections)]
<br>
<br>
<p><a name="est_param"></a></p>
### Parametric Techniques
[[back to top](#sections)]

<p><a name="mle"></a></p>
### Introduction to the Maximum Likelihood Estimate (MLE)
[[back to top](#sections)]
<br>
![](./Images/mle.png)

[View IPython Notebook](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/parameter_estimation_techniques/maximum_likelihood_estimate.ipynb?create=1)
<br>
<p><a name="mle_dist"></a></p>
### Maximum Liklihood parameter Estimation (MLE) for different distributions
[[back to top](#sections)]
<br>
<br>
![](./Images/mle_distributions.png)

[View IPython Notebook](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/parameter_estimation_techniques/max_likelihood_est_distributions.ipynb?create=1)
<br><br>
<br><br>

<p><a name="est_nonparam"></a></p>

### Non-Parametric Techniques
[[back to top](#sections)]

<br>
<br>
<p><a name="parzen"></a></p>
### Kernel density estimation via the Parzen-window technique
[[back to top](#sections)]
<br>
![](./Images/parzen_window_effect.png)

[View IPython Notebook](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/parameter_estimation_techniques/parzen_window_technique.ipynb?create=1)  

[Download PDF](https://github.com/rasbt/pattern_classification/raw/master/PDFs/parzen_window_sebastian_raschka.pdf)
<br>
<br>
<br>
<br>
<br>
<br>
 <p><a name="stat_pat_rec"></a></p>
# Statistical Pattern Recognition
[[back to top](#sections)]

<p><a name="supervised"></a></p>
## Supervised Learning

[[back to top](#sections)]
<p><a name="param"></a></p>
### Parametric Techniques

[[back to top](#sections)]
<p><a name="univar"></a></p>
#### Univariate Normal Density
[[back to top](#sections)]

<hr>
## Example 1
[[back to top](#sections)]

##### Problem Category:
- Statistical Pattern Recognition   
- Supervised Learning  
- Parametric Learning  
- Bayes Decision Theory  
- Univariate data  
- 2-class problem
- equal variances
- equal priors
- Gaussian model (2 parameters)
- No Risk function

![](./Images/1_stat_superv_parametric.png)


[View IPython Notebook](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/stat_pattern_class/supervised/parametric/1_stat_superv_parametric.ipynb?create=1)  
<br>
[Download PDF](https://github.com/rasbt/pattern_classification/raw/master/stat_pattern_class/supervised/parametric/1_stat_superv_parametric.pdf)

<hr>

## Example 2
[[back to top](#sections)]

##### Problem Category:
- Statistical Pattern Recognition   
- Supervised Learning  
- Parametric Learning  
- Bayes Decision Theory  
- Univariate data  
- 2-class problem
- different variances
- equal priors
- Gaussian model (2 parameters)
- No Risk function

![](./Images/2_stat_superv_parametric.png)


[View IPython Notebook](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/stat_pattern_class/supervised/parametric/2_stat_superv_parametric.ipynb?create=1)  
<br>
[Download PDF](https://github.com/rasbt/pattern_classification/raw/master/stat_pattern_class/supervised/parametric/2_stat_superv_parametric.pdf)

<hr>

## Example 3
[[back to top](#sections)]

##### Problem Category:
- Statistical Pattern Recognition   
- Supervised Learning  
- Parametric Learning  
- Bayes Decision Theory  
- Univariate data  
- 2-class problem
- equal variances
- different priors
- Gaussian model (2 parameters)
- No Risk function

![](./Images/3_stat_superv_parametric.png)


[View IPython Notebook](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/stat_pattern_class/supervised/parametric/3_stat_superv_parametric.ipynb?create=1)  
<br>
[Download PDF](https://github.com/rasbt/pattern_classification/raw/master/stat_pattern_class/supervised/parametric/3_stat_superv_parametric.pdf)

<hr>

## Example 4
[[back to top](#sections)]

##### Problem Category:
- Statistical Pattern Recognition   
- Supervised Learning  
- Parametric Learning  
- Bayes Decision Theory  
- Univariate data  
- 2-class problem
- different variances
- different priors
- Gaussian model (2 parameters)
- With conditional Risk (loss functions)

![](./Images/4_stat_superv_parametric.png)


[View IPython Notebook](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/stat_pattern_class/supervised/parametric/4_stat_superv_parametric.ipynb?create=1)  
<br>
[Download PDF](https://github.com/rasbt/pattern_classification/raw/master/stat_pattern_class/supervised/parametric/4_stat_superv_parametric.pdf)

<hr>

## Example 5
[[back to top](#sections)]

##### Problem Category:
- Statistical Pattern Recognition   
- Supervised Learning  
- Parametric Learning  
- Bayes Decision Theory  
- Univariate data  
- 2-class problem
- different variances
- equal priors
- **Cauchy model** (2 parameters)
- With conditional Risk (1-0 loss functions)

![](./Images/6_stat_superv_parametric.png)


[View IPython Notebook](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/stat_pattern_class/supervised/parametric/6_stat_superv_parametric.ipynb?create=1)  
<br>
[Download PDF](https://github.com/rasbt/pattern_classification/raw/master/stat_pattern_class/supervised/parametric/6_stat_superv_parametric.pdf)

<hr>


<p><a name="multivar"></a></p>
#### Multivariate Normal Density

## Example 1
[[back to top](#sections)]

##### Problem Category:
- Statistical Pattern Recognition   
- Supervised Learning  
- Parametric Learning  
- Bayes Decision Theory  
- Multivariate data (2-dimensional)
- 2-class problem
- different variances
- equal prior probabilities
- Gaussian model (2 parameters)
- with conditional Risk (1-0 loss functions)

![](./Images/5_stat_superv_parametric.png)


[View IPython Notebook](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/stat_pattern_class/supervised/parametric/5_stat_superv_parametric.ipynb?create=1)  
<br>
[Download PDF](https://github.com/rasbt/pattern_classification/raw/master/stat_pattern_class/supervised/parametric/5_stat_superv_parametric.pdf)

<hr>


 



