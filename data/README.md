#Table of Contents

- [titanic.csv](#titaniccsv)
- [wine.csv](#winecsv)
- [iris.csv](#iriscsv)




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


