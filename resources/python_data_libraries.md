[[<- back](https://github.com/rasbt/pattern_classification)] to the pattern_classification repository


Sebastian Raschka  
last updated: 09/02/2014

# Useful libraries for data science in Python

This is not meant to be a complete list of all Python libraries out there that are related to scientific computing and data analysis -- printed on paper and stacked one on top of the other, the stack could easily reach a height of 238,857 miles, the distance from Earth to Moon.

However, I would still be looking forward to additions and suggestions.  
Please feel free to drop me a note via
[twitter](https://twitter.com/rasbt), [email](mailto:bluewoodtree@gmail.com), or [google+](https://plus.google.com/+SebastianRaschka).


<br>
<br>


<a class="mk-toclify" id="table-of-contents"></a>

### Table of Contents
- [Fundamental Libraries for Scientific Computing](#fundamental-libraries-for-scientific-computing)
    - [IPython Notebook](#ipython-notebook)
    - [NumPy](#numpy)
    - [pandas](#pandas)
    - [SciPy](#scipy)
- [Math and Statistics](#math-and-statistics)
    - [SymPy](#sympy)
    - [Statsmodels](#statsmodels)
- [Machine Learning](#machine-learning)
    - [Scikit-learn](#scikit-learn)
    - [Shogun](#shogun)
    - [PyBrain](#pybrain)
    - [PyLearn2](#pylearn2)
    - [PyMC](#pymc)
- [Plotting and Visualization](#plotting-and-visualization)
    - [Bokeh](#bokeh)
    - [d3py](#d3py)
    - [ggplot](#ggplot)
    - [matplotlib](#matplotlib)
    - [plotly](#plotly)
    - [prettyplotlib](#prettyplotlib)
    - [seaborn](#seaborn)


<br>
<br>

<a class="mk-toclify" id="fundamental-libraries-for-scientific-computing"></a>
## Fundamental Libraries for Scientific Computing

[back to top](#table-of-contents)

<br>

<a class="mk-toclify" id="ipython-notebook"></a>
#### IPython Notebook

Website: [http://ipython.org/notebook.html](http://ipython.org/notebook.html)

IPython is an alternative Python command line shell for interactive computing with lots of useful enhancements over the "default" Python interpreter.  
The browser-based documents IPython Notebooks are a great environment for scientific computing: Not only to execute code, but also to add informative documentation via Markdown, HTML, LaTeX, embedded images, and inline data plots via e.g., matplotlib. 


<br>

<a class="mk-toclify" id="numpy"></a>
#### NumPy

Website: [http://www.numpy.org](http://www.numpy.org)

NumPy is probably the most fundamental package for efficient scientific computing in Python through linear algebra routines. One of NumPy's major strength is that most operations are implemented as C/C++ and FORTRAN code for efficiency. At its core, NumPy works with multi-dimensional array objects that support broadcasting and lead to efficient, vectorized code.

<br> 
 
<a class="mk-toclify" id="pandas"></a>
#### pandas

Website: [http://pandas.pydata.org](http://pandas.pydata.org)

Pandas is a library for operating with table-like structures. It comes with a powerful DataFrame object, which is a multi-dimensional array object for efficient numerical operations similar to NumPy's *ndarray* with additional functionalities.

<br>
 
<a class="mk-toclify" id="scipy"></a>
#### SciPy

Website: [http://scipy.org/scipylib/index.html](http://scipy.org/scipylib/index.html)

SciPy is a considered to be one of the core packages for scientific computing routines. As a useful expansion of the NumPy core functionality, it contains a broad range of functions for linear algebra, interpolation, integration, clustering, and [many more](http://docs.scipy.org/doc/scipy/reference/index.html).

<br>
<br>

<a class="mk-toclify" id="math-and-statistics"></a>
## Math and Statistics
[back to top](#table-of-contents)


<br>

<a class="mk-toclify" id="sympy"></a>
#### SymPy

Website: [http://www.sympygamma.com](http://www.sympygamma.com)

SymPy is a Python library for symbolic mathematical computations. It has a broad range of features, ranging from calculus, algebra, geometry, discrete mathematics, and even quantum physics. It also includes basic plotting functionality and print functions with LaTeX support.

<br>

<a class="mk-toclify" id="statsmodels"></a>
#### Statsmodels

Website: [http://statsmodels.sourceforge.net](http://statsmodels.sourceforge.net)

Statsmodel is a Python libarary that is centered around statistical data analysis mainly through linear models and includes a variety of statistical tests.

<br>
<br>


<a class="mk-toclify" id="machine-learning"></a>
## Machine Learning
[back to top](#table-of-contents)


<br>

<a class="mk-toclify" id="scikit-learn"></a>
#### Scikit-learn

Website: [http://scikit-learn.org/stable/](http://scikit-learn.org/stable/)

Scikit-learn is is probably the most popular general machine library for Python. It includes a broad range of different classifiers, cross-validation and other model selection methods, dimensionality reduction techniques, modules for regression and clustering analysis, and a useful data-preprocessing module.

<br>

<a class="mk-toclify" id="shogun"></a>
#### Shogun

Website: [http://www.shogun-toolbox.org](http://www.shogun-toolbox.org)

Shogun is a machine learning library that is focussed on large-scale kernel methods. Its particular strengths are Support Vector Machines (SVMs) and it comes with a range of different SVM implementations.

<br>

<a class="mk-toclify" id="pybrain"></a>
#### PyBrain

Website: [http://pybrain.org](http://pybrain.org)

<br>

PyBrain (Python-Based Reinforcement Learning, Artificial Intelligence and Neural Network Library) is a machine learning library that uses neural networks to focus on supervised learning, reinforcement learning, and evolutionary methods.

<br>

<a class="mk-toclify" id="pylearn2"></a>
#### PyLearn2

Website: [http://deeplearning.net/software/pylearn2/](http://deeplearning.net/software/pylearn2/)

PyLearn2 is a machine learning **research** library - a library to study machine learning - focussed on convolutional neural networks.

<br>

<a class="mk-toclify" id="pymc"></a>
#### PyMC

Website: [http://pymc-devs.github.io/pymc/index.html](http://pymc-devs.github.io/pymc/index.html)

The focus of PyMC is Bayesian statistics and comes with a broad range of algorithms (including Markov Chain Monte Carlo, MCMC) for model fitting.

<br>
<br>

<a class="mk-toclify" id="plotting-and-visualization"></a>
## Plotting and Visualization
[back to top](#table-of-contents)


<br>


<a class="mk-toclify" id="bokeh"></a>
#### Bokeh

Website: [http://bokeh.pydata.org](http://bokeh.pydata.org)

Bokeh is a plottling library that is focussed on aesthetic layouts and interactivity to produce high-quality plots for web browsers.

<br>

<a class="mk-toclify" id="d3py"></a>
#### d3py

Website: [https://github.com/mikedewar/d3py](https://github.com/mikedewar/d3py)

d3py is a plotting library to create interactive data visualizations based on d3. 


<br>

<a class="mk-toclify" id="ggplot"></a>
#### ggplot

Website: [https://github.com/yhat/ggplot](https://github.com/yhat/ggplot)

ggplot is a port of R's popular ggplot2 library, which brings the alternative syntax and unique visualization style to Python. 

<br>

<a class="mk-toclify" id="matplotlib"></a>
#### matplotlib

Website: [http://matplotlib.org](http://matplotlib.org)

Matplotlib is Python's most popular and comprehensive plotting library that is especially useful in combination with NumPy/SciPy. 

<br>

<a class="mk-toclify" id="plotly"></a>
#### plotly

Website: [https://plot.ly](https://plot.ly)

Plotly is a plotting library that is focussed on adding interactivity to data visualizations and to share them via the web for collaborative data analysis.

<br>

<a class="mk-toclify" id="prettyplotlib"></a>
#### prettyplotlib

Website: [http://olgabot.github.io/prettyplotlib/](http://olgabot.github.io/prettyplotlib/)

Prettyplotlib is a nice enhancement-library that turns matplotlib's default styles into beautiful, presentation-ready plots based on information design and color perception studies.

<br>

<a class="mk-toclify" id="seaborn"></a>
#### seaborn

Website: [http://web.stanford.edu/~mwaskom/software/seaborn/](http://web.stanford.edu/~mwaskom/software/seaborn/)

Seaborn is based on matplotlib's core functionality and adds additional features (e.g., violin plots) and visual enhancements to create even more beautiful plots.


