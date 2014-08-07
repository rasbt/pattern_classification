Sebastian Raschka  
Last updated 09/06/2014

# General tips and advices


<br>

### A typical pattern classification workflow



<hr> 
(1) question -> (2) input data -> (3) features -> (4) algorithm -> (5) parameters -> (6) evaluation 
 <hr>

- The question should be concrete and specific: "Can I specify flower species based on the dimensions of the leaves?".

- Usually: the more input data is collected the better.

- Try to consult a domain expert for selecting "good" features.

- choice of a learning and prediction algorithm is typically less important than good feature selection: 
	- Hand, David J. 2006. “Classifier Technology and the Illusion of Progress.” Statistical Science 21 (1): 1–14. doi:10.1214/088342306000000060. 
[http://projecteuclid.org/euclid.ss/1149600839](http://projecteuclid.org/euclid.ss/1149600839)

- It is expected that the generalization error (error on the test data set, or any other new dataset that was not used for training/fitting the model) is larger than the resubstitution error (= the error on the training dataset). However a large difference between the two error rates is a strong indicator for overfitting.

### in progress ...