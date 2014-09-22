# The effect of scaling and mean centering of variables prior to a Principal Component Analysis

Let us think about whether it matters or not if the variables are centered for applications such as Principal Component Analysis (PCA) if the PCA is calculated from the covariance matrix (i.e., the \\(k\\) principal components are the eigenvectors of the covariance matrix that correspond to the \\(k\\) largest eigenvalues.


<br>

### 1. Mean centering does not affect the covariance matrix

Here, the rational is: If the covariance is the same whether the variables are centered or not, the result of the PCA will be the same.

Let's assume we have the 2 variables \\(\bf{x}\\) and \\(\bf{y}\\) Then the covariance between the attributes is calculated as

\\[ \sigma_{xy} = \frac{1}{n-1} \sum_{i}^{n} (x_i - \bar{x})(y_i - \bar{y})   \\]

Let us write the centered variables as 

\\[ x' = x - \bar{x} \text{ and } y' = y - \bar{y} \\]

The centered covariance would then be calculated as follows:

\\[ \sigma_{xy}' = \frac{1}{n-1} \sum_{i}^{n} (x_i' - \bar{x}')(y_i' - \bar{y}')   \\]

But since after centering, \\(\bar{x}' = 0\\) and \\(\bar{y}' = 0\\) we have 

\\[ \sigma_{xy}' = \frac{1}{n-1} \sum_{i}^{n} x_i' y_i'   \\] which is our original covariance matrix if we resubstitute back the terms 
\\[ x' = x - \bar{x} \text{ and } y' = y - \bar{y} \\].

Even centering only one variable, e.g., \\(\bf{x}\\) wouldn't affect the covariance:

\\[ \sigma_{\text{xy}} = \frac{1}{n-1} \sum_{i}^{n} (x_i' - \bar{x}')(y_i - \bar{y})   \\]
\\[  =  \frac{1}{n-1} \sum_{i}^{n} (x_i' - 0)(y_i - \bar{y})   \\]
\\[  =  \frac{1}{n-1} \sum_{i}^{n} (x_i - \bar{x})(y_i - \bar{y})   \\]

<br>

### 2. Scaling of variables does affect the covariance matrix

If one variable is scaled, e.g, from pounds into kilogram (1 pound = 0.453592 kg), it does affect the covariance and therefore influences the results of a PCA.


Let \\(c\\) be the scaling factor for \\(\bf{x}\\)

Given that the "original" covariance is calculated as

\\[ \sigma_{xy} = \frac{1}{n-1} \sum_{i}^{n} (x_i - \bar{x})(y_i - \bar{y})   \\]

the covariance after scaling would be calculated as:

\\[ \sigma_{xy}' = \frac{1}{n-1} \sum_{i}^{n} (c \cdot x_i - c \cdot  \bar{x})(y_i - \bar{y})   \\]
\\[ =  \frac{c}{n-1} \sum_{i}^{n} (x_i -   \bar{x})(y_i - \bar{y})   \\]

\\[ \Rightarrow \sigma_{xy} = \frac{\sigma_{xy}'}{c} \\]
\\[ \Rightarrow \sigma_{xy}' = c \cdot \sigma_{xy} \\]

Therefore, the covariance after scaling one attribute by the constant \\(c\\) will result in a rescaled covariance \\(c \sigma_{xy}\\) So if we'd scaled \\(\bf{x}\\) from pounds to kilograms, the covariance between \\(\bf{x}\\) and \\(\bf{y}\\) will be 0.453592 times smaller.

<br>

### 3. Standardizing affects the covariance

Standardization of features will have an effect on the outcome of a PCA (assuming that the variables are originally not standardized). This is because we are scaling the covariance between every pair of variables by the product of the standard deviations of each pair of variables.

The equation for standardization of a variable is written as 

\\[ z = \frac{x_i - \bar{x}}{\sigma} \\]

The "original" covariance matrix:

\\[ \sigma_{xy} = \frac{1}{n-1} \sum_{i}^{n} (x_i - \bar{x})(y_i - \bar{y})   \\]

And after standardizing both variables:

\\[ x' = \frac{x - \bar{x}}{\sigma_x} \text{ and } y' =\frac{y - \bar{y}}{\sigma_y} \\]


\\[ \sigma_{xy}' =  \frac{1}{n-1} \sum_{i}^{n} (x_i' - 0)(y_i' - 0)   \\]

\\[  =  \frac{1}{n-1} \sum_{i}^{n} \bigg(\frac{x - \bar{x}}{\sigma_x}\bigg)\bigg(\frac{y - \bar{y}}{\sigma_y}\bigg)   \\]

\\[   = \frac{1}{(n-1) \cdot \sigma_x \sigma_y} \sum_{i}^{n} (x_i - \bar{x})(y_i - \bar{y})   \\]

\\[ \Rightarrow \sigma_{xy}' = \frac{\sigma_{xy}}{\sigma_x \sigma_y} \\]

