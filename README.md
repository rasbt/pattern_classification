pattern_classification
======================

Examples for solving pattern classification problems in Python (IPython Notebooks)

<br>
<br>
# Sections
&#8226; <a href="#stat_pat_rec"><strong>Statistical Pattern Recognition</strong></a><br>
&nbsp;&nbsp;&nbsp;&#8226; <a href="#supervised">Supervised Learning</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; <a href="#param">Parametric Techniques</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; <a href="#univar">Univariate Normal Density</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; <a href="#multivar">Multivariate Normal Density</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; Non-Parametric Techniques<br>
<br>
&nbsp;&nbsp;&nbsp;&#8226; Unsupervised Learning<br>
<br>
&nbsp;&#8226; <a href="#dim_red"><strong>Techniques for Dimensionality Reduction</strong></a><br>
&nbsp;&nbsp;&nbsp;&#8226; <a href="#feat_sel">Feature Selection</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; <a href="#seq_feat_sel_algos">Sequential Feature Selection Algorithms</a><br>
&nbsp;&nbsp;&nbsp;&#8226; <a href="#projection">Projection</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; <a href="#comp_analysis">Component Analyses</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; <a href="#linear_transf">Linear Transformation</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; <a href="#pca">Principal Component Analysis (PCA)</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8226; <a href="#mda">Multiple Discriminant Analysis (MDA)</a><br>
<hr>




<br>
<br>


<p><a name="stat_pat_rec"></a></p>
# Statistical Pattern Recognition

<p><a name="supervised"></a></p>
## Supervised Learning

<p><a name="param"></a></p>
### Parametric Approach

<p><a name="univar"></a></p>
#### Univariate Normal Density


<hr>
## Example 1

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




<p><a name="dim_red"></a></p>

#Techniques for Dimensionality Reduction

<p><a name="feat_sel"></a></p>

## Feature Selection

<p><a name="seq_feat_sel_algos"></a></p>

#### Sequential Feature Selection Algorithms
![](./Images/feat_sele_alg.png)

[View IPython Notebook](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/dimensionality_reduction/feature_selection/sequential_selection_algorithms.ipynb?create=1)  
<br>
[Download PDF](https://github.com/rasbt/pattern_classification/tree/master/dimensionality_reduction/feature_selection/sequential_selection_algorithms.pdf)

<hr>

<p><a name="projection"></a></p>

## Projection

<p><a name="comp_analyses"></a></p>

### Component Analyses

<p><a name="linear_transf"></a></p>

### Linear Transformation

<br>
<br>


<p><a name="pca"></a></p>

#### Principal Component Analyses (PCA)
<br><br>
![./Images/principal_component_analysis.png](./Images/principal_component_analysis.png)
<br><br>
[View IPython Notebook](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/dimensionality_reduction/projection/principal_component_analysis.ipynb?create=1)  

<p><a name="mda"></a></p>
<br>
<br>

#### Multiple Discriminant Analysis (MDA)