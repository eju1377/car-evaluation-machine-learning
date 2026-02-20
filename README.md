# car-evaluation-machine-learning
Machine learning on the Car Evaluation dataset for Machine Learning final project.

## About the Data
This dataset comes from the UC Irvine Machine Learning Repository and contains categorical data about cars.

[Car Evaluation](https://archive.ics.uci.edu/dataset/19/car+evaluation) contains 7 columns, which are:
1. `buying` - Buying price (vhigh, high, med, low)
2. `maint` - Price of the maintenance (vhigh, high, med, low)
3. `doors` - Number of doors (2, 3, 4, 5more)
4. `persons` - Capacity in terms of persons to carry (2, 4, more)
5. `lug_boot` - The size of luggage boot (small, med, big)
6. `safety` - Estimated safety of the car (low, med, high)
7. `class` - Evaluation level (unacc, acc, good, vgood)

The problem I am trying to solve is a classification problem where the target variable is `class` and the features are `buying`, `maint`, `doors`, `persons`, `lug_boot`, and `safety`.

## Set Up and Preprocessing

### EDA
I imported the dataset directly from the repository and converted the data into a DataFrame. I created a profile report to explore the data. I found that the values in `class` are unbalanced and that the `class` and `safety` columns have the highest correlation.

![Distribution of Class](/charts/class.png)

### Preprocessing
I found no duplicates or null values. Instead of using `LabelEncoder`, I manually encoded the categorical variables to avoid incorrect ordering. This was important because all features have a logical relationship instead of an alphabetical one. Then I split the data into train and test sets, using 25% of the data in the test set.

## Model Implementation and Evaluation

### Entropy Model Implementation and Evaluation
The goal for this model is to predict the class of a car based on given features. I instantiated a decision tree with entropy criteria using `DecisionTreeClassifier`. I fit the model on the training data and predicted results using the test data. Then I created a diagram showing the decision tree, a confusion matrix, and printed a classification report. The accuracy of the model is well above the baseline accuracy of the dataset. This most likely means that this is a strong model, but could also mean that the model has been overfitted to this particular dataset.

![Entropy Model Confusion Matrix](/charts/entropy_cm.png)

### Gini Model Implementation and Evaluation
The goal for this model is to predict the class of a car based on given features. I instantiated a decision tree with gini criteria using `DecisionTreeClassifier`. I fit the model on the training data and predicted results using the test data. Then I created a diagram showing the decision tree, a confusion matrix, and printed a classification report. The results are practically identical to the entropy model. There is no significant advantage to using this model except computational efficiency.

![Gini Model Confusion Matrix](/charts/gini_cm.png)

### Random Forest Model Implementation and Evaluation
The goal for this model is to predict the class of a car based on given features. Using a random forest model will remove any overfitting and generalize better than a single decision tree. I instantiated a random forest model with entropy criteria using `RandomForestClassifier`. I fit the model on the training data and predicted results using the test data. Then I created a confusion matrix and printed a classification report. The random forest model predicted better than the decision tree models. The incorrect predictions were evenly spread between different labels. I think this is better than having the incorrect predictions clustered near one label.

![Random Forest Model Confusion Matrix](/charts/rf_cm.png)

## Reflection
The main challenge in this project came with encoding variables. When I was encoding the categorical variables, I first used `LabelEncoder` which encodes the variables in alphabetical order. This was a problem because the variables have an order that is not alphabetical. I learned how to use `.map()` to manually encode with a chosen order. I learned that proper encoded is critical in machine learning. I also learned that ensemble models improve robustness compared to singular models.
