[Sebastian Raschka](http://sebastianraschka.com)

Last updated 07/12/2014

<hr>

I frequently embed all sorts of equations in my IPython notebooks, but instead of re-typing them every time, I thought that it might be worthwhile to have a copy&paste-ready equation glossary at hand.

Since I recently had to work without internet connection, I decided compose this in a MathJax-free manner.

For example, if you want to use those equations in a IPython notebook markdown cell, simply 
Y$-signs, e.g., 

`$\mu = 2$`    

or prepend `/begin{equation}` and append `/end{equation}`

![](../Images/latex_equations/ipython_example_1.png)

<hr>

<br>
<br>
<br>
<br>



### Bayes Theorem

- Naive Bayes' classifier:

	- posterior probability:

	![](../Images/latex_equations/bayes_classifier_1.gif)

		P(\omega_j|x) = \frac{p(x|\omega_j) \cdot P(\omega_j)}{p(x)}
		
	![](../Images/latex_equations/bayes_theorem_words.gif)	
		
		\Rightarrow posterior \; probability = \frac{\; likelihood \; \cdot \; prior \; probability}{evidence}
 
 
	- decision rule:

	![](../Images/latex_equations/bayes_decision_rule_1.gif)

	![](../Images/latex_equations/bayes_decision_rule_2.gif)
    
        Decide \;  \omega_1 $ if  P(\omega_1|x) > P(\omega_2|x) \; else \; decide \; \omega_2 .
		
		\frac{p(x|\omega_1) \cdot P(\omega_1)}{p(x)} > \frac{p(x|\omega_2) \cdot P(\omega_2)}{p(x)}

	- objective functions:
	
	![](../Images/latex_equations/bayes_objective_function_1.gif)
	
		g_1(\pmb x) = P(\omega_1 | \; \pmb{x}), \quad  g_2(\pmb{x}) = P(\omega_2 | \; \pmb{x}), \quad  g_3(\pmb{x}) = P(\omega_2 | \; \pmb{x})

	![](../Images/latex_equations/bayes_objective_function_2.gif)

		\quad g_i(\pmb{x}) = \pmb{x}^{\,t} \bigg( - \frac{1}{2} \Sigma_i^{-1} \bigg) \pmb{x} + \bigg( \Sigma_i^{-1} \pmb{\mu}_{\,i}\bigg)^t \pmb x + \bigg( -\frac{1}{2} \pmb{\mu}_{\,i}^{\,t}  \Sigma_{i}^{-1} \pmb{\mu}_{\,i} -\frac{1}{2} ln(|\Sigma_i|)\bigg)


<br>
<br>

### Binomial distribution

- Probability density function:

![](../Images/latex_equations/binomial_distribution.gif)

	p_k = \bigg[ \begin{array}{c} n \\ k \end{array}\bigg] \cdot p^k \cdot (1-p)^{n-k}


<br>
<br>

### Co-Variance

![](../Images/latex_equations/covariance.gif)

	S_{xy} = \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})\quad

example covariance matrix:

![](../Images/latex_equations/covariance_matrix.gif)

	\quad\pmb{\Sigma_1} = 
	\begin{bmatrix}1\quad 0\quad 0\\0\quad 1\quad0\\0\quad0\quad1\end{bmatrix}\quad

<br>
<br>

### Eigenvector and Eigenvalue

![](../Images/latex_equations/eigenvector_equation.gif)

    \pmb A\pmb{v} =  \lambda\pmb{v}\\\\

    where \\\\

    \pmb A = S_{W}^{-1}S_B\\
    \pmb{v} = \; Eigenvector\\
    \lambda = \; Eigenvalue
    
   

<br>
<br>

### Least-squares fit regression

- Linear equation

![](../Images/latex_equations/least_squares_linear_equation.gif)

	f(x) = a\cdot x + b
	
Slope:

![](../Images/latex_equations/least_squares_slope.gif)

	a = \frac{S_{x,y}}{\sigma_{x}^{2}}\quad

Y-axis intercept:

![](../Images/latex_equations/least_squares_y-intercept.gif)

	b = \bar{y} - a\bar{x}\quad
	
where

![](../Images/latex_equations/least_squares_variance_covariance.gif)

	S_{xy} = \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})\quad (covariance) \\
	\sigma{_x}^{2} = \sum_{i=1}^{n} (x_i - \bar{x})^2\quad (variance)	
	
<br>
<br>	
	
- Matrix equation

![](../Images/latex_equations/least_squares_matrix_equation_1.gif)

	\pmb X \; \pmb a = \pmb y
	
![](../Images/latex_equations/least_squares_matrix_equation_2.gif)

	\pmb X \; \pmb a = \pmb y

    \Bigg[ \begin{array}{cc}
    x_1 & 1  \\
    ... & 1 \\
    x_n & 1  \end{array} \Bigg]$
    $\bigg[ \begin{array}{c}
    a  \\
    b \end{array} \bigg]$
    $=\Bigg[ \begin{array}{c}
    y_1   \\
    ...  \\
    y_n  \end{array} \Bigg]


![](../Images/latex_equations/least_squares_matrix_equation_3.gif)

	\pmb a = (\pmb X^T \; \pmb X)^{-1} \pmb X^T \; \pmb y


<br>
<br>



### Linear Discriminant Analysis

- In-between class scatter matrix

![](../Images/latex_equations/lda_scatter_matrix.gif)


    S_W = \sum\limits_{i=1}^{c} S_i \\\\

    where  \\\\

    S_i = \sum\limits_{\pmb x \in D_i}^n (\pmb x - \pmb m_i)\;(\pmb x - \pmb m_i)^T
    (scatter \; matrix \; for \; every \; class) \\\\

    and \\\\
      
    \pmb m_i = \frac{1}{n_i} \sum\limits_{\pmb x \in D_i}^n \; \pmb x_k \;(mean \; vector)m	\limits_{\pmb x \in D_i}^n \; \pmb x_k \;(mean \; vector)

<br>
<br>

- Between class scatter matrix

![](../Images/latex_equations/lda_between_class_scattermatrix.gif)
	
	S_B = \sum\limits_{i=1}^{c} (\pmb m_i - \pmb m) (\pmb m_i - \pmb m)^T

<br>
<br>


### Maximum Likelihood Estimate

The probability of observing the data set 

![](../Images/latex_equations/mle_data.gif)

	D = \left\{ \pmb x_1, \pmb x_2,..., \pmb x_n \right\} 
	
 can be pictured as probability to observe a particular sequence of patterns,  
where the probability of observing a particular patterns depends on **&theta;**, the parameters the underlying (class-conditional) distribution. In order to apply MLE, we have to make the assumption that the samples are *i.i.d.* (independent and identically distributed).

![](../Images/latex_equations/mle_likelihood.gif)

    p(D\; | \;  \pmb \theta\;) \\\\
    = p(\pmb x_1 \; | \; \pmb \theta\;)\; \cdot \; p(\pmb x_2 \; | \;\pmb \theta\;) \; \cdot \;...  \; p(\pmb x_n \; | \; \pmb \theta\;) \\\\
    = \prod_{k=1}^{n} \; p(\pmb x_k \pmb \; | \; \pmb \theta \;)


Where **&theta;** is the parameter vector, that contains the parameters for a particular distribution that we want to estimate.

and p(D | **&theta;**) is also called the likelihood of **&theta;**.

- log-likelihood
 
![](../Images/latex_equations/mle_log_likelihood.gif)


	p(D|\theta) = \prod_{k=1}^{n} p(x_k|\theta) \\
	\Rightarrow l(\theta) = \sum_{k=1}^{n} ln \; p(x_k|\theta)


- Differentiation

![](../Images/latex_equations/mle_nabla_1.gif)

    \nabla_{\pmb \theta} \equiv \begin{bmatrix}  
    \frac{\partial \; }{\partial \; \theta_1} \\
    \frac{\partial \; }{\partial \; \theta_2} \\
    ...\\
    \frac{\partial \; }{\partial \; \theta_p}\end{bmatrix}


![](../Images/latex_equations/mle_nabla_2.gif)

    \nabla_{\pmb \theta} l(\pmb\theta) \equiv \begin{bmatrix}  
    \frac{\partial \; L(\pmb\theta)}{\partial \; \theta_1} \\
    \frac{\partial \; L(\pmb\theta)}{\partial \; \theta_2} \\
    ...\\
    \frac{\partial \; L(\pmb\theta)}{\partial \; \theta_p}\end{bmatrix}$
    $= \begin{bmatrix}  
    0 \\
    0 \\
    ...\\
    0\end{bmatrix}

- parameter vector

![](../Images/latex_equations/mle_parameter_vector.gif)

    \pmb \theta_i = \bigg[ \begin{array}{c}
    \ \theta_{i1} \\
    \ \theta_{i2} \\
    \end{array} \bigg]=
    \bigg[ \begin{array}{c}
    \pmb \mu_i \\
    \pmb \Sigma_i \\
    \end{array} \bigg]


<br>
<br>

### Min-Max scaling

![](../Images/latex_equations/minmax_scaling.gif)

	X_{norm} = \frac{X - X_{min}}{X_{max}-X_{min}}

<br>
<br>

### Normal distribution (multivariate)

- Probability density function

![](../Images/latex_equations/gaussian_multivariate.gif)

	p(\pmb x) \sim N(\pmb \mu|\Sigma)\\\\

	p(\pmb x) \sim \frac{1}{(2\pi)^{d/2} \; |\Sigma|^{1/2}} exp \bigg[ -\frac{1}{2}(\pmb x - \pmb \mu)^t \Sigma^{-1}(\pmb x - \pmb \mu) \bigg]

<br>
<br>

### Normal distribution (univariate)

- Probability density function

![](../Images/latex_equations/gaussian_pdf_univariate.gif)


	p(x) \sim N(\mu|\sigma^2) \\\\

	p(x) \sim \frac{1}{\sqrt{2\pi\sigma^2}} \exp{ \bigg[-\frac{1}{2}\bigg( \frac{x-\mu}{\sigma}\bigg)^2 \bigg] } $

<br>
<br>

### Parzen window function

![](../Images/latex_equations/parzen_window_function_hypercube_1.gif)

	\phi(\pmb u) = \Bigg[ \begin{array}{ll} 1 & \quad |u_j| \leq 1/2 \; ;\quad \quad j = 1, ..., d \\
	0 & \quad otherwise \end{array} 

for a hypercube of unit length 1 centered at the coordinate system's origin. What this function basically does is assigning a value 1 to a sample point if it lies within 1/2 of the edges of the hypercube, and 0 if lies outside (note that the evaluation is done for all dimensions of the sample point).

If we extend on this concept, we can define a more general equation that applies to hypercubes of any length *h<sub>n</sub>* that are centered at ***x***: 

![](../Images/latex_equations/parzen_window_function_hypercube_2.gif)

	k_n = \sum\limits_{i=1}^{n} \phi \bigg( \frac{\pmb x - \pmb x_i}{h_n} \bigg)\\\\

	where \; \pmb u = \bigg( \frac{\pmb x - \pmb x_i}{h_n} \bigg)

- probability density estimation with hypercube kernel

![](../Images/latex_equations/parzen_estimation_hypercube_1.gif)

	p_n(\pmb x) = \frac{1}{n} \sum\limits_{i=1}^{n} \frac{1}{h^d} \phi \bigg[ \frac{\pmb x - \pmb x_i}{h_n} \bigg]

![](../Images/latex_equations/parzen_estimation_hypercube_2.gif)

	where\\\\   
	h^d = V_n\quad   and    \quad\phi \bigg[ \frac{\pmb x - \pmb x_i}{h_n} \bigg] = k

- probability density estimation with Gaussian kernel

![](../Images/latex_equations/parzen_estimation_gaussian_1.gif)

	p_n(\pmb x) = \frac{1}{n} \sum\limits_{i=1}^{n} \frac{1}{h^d} \phi \Bigg[ \frac{1}{(\sqrt {2 \pi})^d h_{n}^{d}} exp \; \bigg[ -\frac{1}{2} \bigg(\frac{\pmb x - \pmb x_i}{h_n} \bigg)^2 \bigg] \Bigg]


<br>
<br>

### Population mean

![](../Images/latex_equations/population_mean.gif)

	\mu = \frac{1}{N} \sum_{i=1}^N x_i

example mean vector:

![](../Images/latex_equations/mean_vector_example.gif)

	\pmb{\mu_1} = 
	\begin{bmatrix}0\\0\\0\end{bmatrix}


<br>
<br>



### Poisson distribution (univariate)

- Probability density function

![](../Images/latex_equations/poisson_univariate.gif)

	p(x|\theta) = \frac{e^{-\theta}\theta^{xk}}{x_k!}


<br>
<br>



### Principal Component Analysis

- Scatter matrix

![](../Images/latex_equations/pca_scatter_matrix.gif)

	S = \sum\limits_{k=1}^n (\pmb x_k - \pmb m)\;(\pmb x_k - \pmb m)^T
	
	
where 

![](../Images/latex_equations/pca_mean_vector.gif)

	\pmb m = \frac{1}{n} \sum\limits_{k=1}^n \; \pmb x_k \; (mean \; vector)

<br>
<br>

### Rayleigh distribution (univariate)

- Probability density function

![](../Images/latex_equations/rayleigh_univariate.gif)

    p(x|\theta) =  \Bigg\{ \begin{array}{c}
      2\theta xe^{- \theta x^2},\quad \quad x \geq0, \\
      0,\quad otherwise. \\
      \end{array}

<br>
<br>

### Standard deviation

	\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2}
	
![](../Images/latex_equations/standard_deviation.gif)

<br>
<br>

### Variance

![](../Images/latex_equations/variance.gif)

	\sigma{_x}^{2} = \sum_{i=1}^{n} (x_i - \bar{x})^2\quad

<br>
<br>

### Z-score

![](../Images/latex_equations/z_score.gif)

 	z = \frac{x - \mu}{\sigma}
 	
<br>
<br>

