#Table of Contents

- [titanic.csv](#titaniccsv)
- [wine.csv](#winecsv)
- [iris.csv](#iriscsv)
- [sms_spam_collection.csv](#sms_spam_collection.csv)
- [50k_imdb_movie_reviews.csv](#50k_imdb_movie_reviews.csv)




<br>
<br>

## titanic.csv
[[back to top](#table-of-contents)]

Source: [http://lib.stat.cmu.edu/S/Harrell/data/descriptions/titanic.html](http://lib.stat.cmu.edu/S/Harrell/data/descriptions/titanic.html)

- sample size: 1313   
- features: 10   

| Name      | Levels | Storage   | NAs |
|-----------|--------|-----------|-----|
| pclass    | 3      | integer   | 0   |
| survived  |        | double    | 0   |
| name      |        | character | 0   |
| age       |        | double    | 680 |
| embarked  | 3      | integer   | 492 |
| home.dest | 371    | integer   | 559 |
| room      |        | character | 0   |
| ticket    |        | character | 0   |
| boat      | 99     | integer   | 966 |
| sex       | 2      | integer   | 0   |


These data were obtained from Robert Dawson, Saint Mary's University, E-mail. The variables are pclass, age, sex, survived. These data frames are useful for demonstrating many of the functions in Hmisc as well as demonstrating binary logistic regression analysis using the Design library. For more details and references see Simonoff, Jeffrey S (1997): The "unusual episode" and a second statistics course. J Statistics Education, Vol. 5 No. 1.

<br>
<hr>
<br>


## wine.csv
[[back to top](#table-of-contents)]

Source: [https://archive.ics.uci.edu/ml/datasets/Wine](https://archive.ics.uci.edu/ml/datasets/Wine)

Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.



|wine.csv.csv					  |		  			|
|----------------------------|----------------|
| Samples                    | 178            |
| Features                   | 13             |
| Classes                    | 3              |
| Data Set Characteristics:  | Multivariate   |
| Attribute Characteristics: | Integer, Real  |
| Associated Tasks:          | Classification |
| Missing Values             | None           |

|	column| attribute	|
|-----|------------------------------|
| 1)  | Class Label                  |
| 2)  | Alcohol                      |
| 3)  | Malic acid                   |
| 4)  | Ash                          |
| 5)  | Alcalinity of ash            |
| 6)  | Magnesium                    |
| 7)  | Total phenols                |
| 8)  | Flavanoids                   |
| 9)  | Nonflavanoid phenols         |
| 10) | Proanthocyanins              |
| 11) | intensity                    |
| 12) | Hue                          |
| 13) | OD280/OD315 of diluted wines |
| 14) | Proline                      |


| class | samples   |
|-------|----|
| 1     | 59 |
| 2     | 71 |
| 3     | 48 |

Original Owners: 

Forina, M. et al, PARVUS - 
An Extendible Package for Data Exploration, Classification and Correlation. 
Institute of Pharmaceutical and Food Analysis and Technologies, Via Brigata Salerno, 
16147 Genoa, Italy. 

<br>
<hr>
<br>

## iris.csv
[[back to top](#table-of-contents)]


Source:[https://archive.ics.uci.edu/ml/datasets/Iris](https://archive.ics.uci.edu/ml/datasets/Iris) 

Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.

|iris.csv					  |		  			|
|----------------------------|----------------|
| Samples                    | 150            |
| Features                   | 4              |
| Classes                    | 3              |
| Data Set Characteristics:  | Multivariate   |
| Attribute Characteristics: | Real           |
| Associated Tasks:          | Classification |
| Missing Values             | None           |


|	column| attribute	|
|-----|------------------------------|
| 1)  | sepal length in cm                  |
| 2)  | sepal width in cm                      |
| 3)  | petal length in cm                   |
| 4)  | petal width in cm                        |
| 5)  | class label|


| class | samples   |
|-------|----|
| Iris-setosa     | 50 |
| Iris-versicolor     | 50 |
| Iris-virginica     | 50 |


Creator: R.A. Fisher (1936)

#### iris_test40.csv and iris_train60.csv



The datasets `iris_test40.csv` and `iris_train60.csv` are randomly sampled from the original `iris.csv` dataset where `iris_test40.csv` contains 40% of the original samples and the remaining 60% are contained in `iris_train60.csv`. The class labels in the 5th column where replaced by integers 1-4, where 

- 1 = Iris-setosa
- 2 = Iris-versicolor
- 3 = Iris-virginica


| class | # of samples in test set | # of samples in training set | total
|-------|------|----------|
| Iris-setosa     | 21 | 29 | 50 |
| Iris-versicolor | 22 | 28 | 50 |
| Iris-virginica  | 17 | 33 | 50 |
| total           | 60 | 90 | 150|


<br>
<hr>
<br>

## sms_spam_collection.csv
[[back to top](#table-of-contents)]

A public dataset of 5572 SMS messages that are labeled as either "spam" or "ham" (not spam).

| 0    | 1    |                                                   |
|------|------|---------------------------------------------------|
| ...  | ...  | ... |
| 5567 | spam | This is the 2nd time we have tried 2 contact u... |
| 5568 | ham  | Will ü b going to esplanade fr home?              |
| 5569 | ham  | Pity, * was in mood for that. So...any other s... |
| 5570 | ham  | The guy did some bitching but I acted like i'd... |
| 5571 | ham  | Rofl. Its true to its name                        |


Source: [http://www.dt.fee.unicamp.br/%7Etiago/smsspamcollection/](http://www.dt.fee.unicamp.br/%7Etiago/smsspamcollection/)

Almeida, Tiago A., José María G. Hidalgo, and Akebo Yamakami. 2011. “Contributions to the Study of SMS Spam Filtering: New Collection and Results.” In Proceedings of the 11th ACM Symposium on Document Engineering, 259–62. DocEng ’11. New York, NY, USA: ACM. doi:10.1145/2034691.2034742.


<br>
<hr>
<br>

## 50k_imdb_movie_reviews.csv
[[back to top](#table-of-contents)]

A CSV file assembled from the 50k IMDb movie review dataset. The dataset consists
of 50,000 movie reviews from the original "train" and "test" subdirectories. The class labels are binary (`1`=positive and `0`=negative) and contain 25,000 positive and 25,000 negative movie reviews, respectively.

| review    | sentiment    | set           |
|------|------|---------------------------------------------------|
| ...  | ...  | ... |
|I saw 'Descent' last night at the Stockholm Fi... | 0 | train
|Some films that you pick up for a pound turn o... | 0 | train
|This is one of the dumbest films, I've ever se... | 0 | train

Source: [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/)

> AL Maas, RE Daly, PT Pham, D Huang, AY Ng, and C Potts. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Lin- guistics: Human Language Technologies, pages 142–150, Portland, Oregon, USA, June 2011. Association for Computational Linguistics.
