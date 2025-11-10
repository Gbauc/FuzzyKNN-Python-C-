An implementation of FuzzyKNN in C++ with Python integration, based on the reference [A fuzzy K-nearest neighbors](https://ieeexplore.ieee.org/document/6313426).

Note that this implementation only works on Linux systems. The C++ code was converted to a Python module using g++ on a Linux system, generating a `.so` file. To run this application on a Windows system, you must recompile the source code in a Windows environment using a C++ compiler of your choice and install pybind11 on your machine to generate the Python module.

First, place the `.so` and `.py` files in the same folder. Then, you can start using the module by importing it:

`from KNN_Fuzzy.py import FuzzyKNN`

Instanciate the class, follows the same logic of scikit-learn models:

```
#Here I'm instanciating the class with k = 3 and m = 2
knn = FuzzyKNN(3,2)
```

Now, you can use `.fit()` and `.predict()` to your data and get the results:

```
#Fitting to training data
knn.fit(x_train, y_train)
#Prediciting in test data
knn.predict(x_test)
```

The `.fit()` function will store the training data in memory and compute the fuzzy membership to each class of your classification problem to each training point. While the `.predict()` fuction will compute the fuzzy membership of the test data to each classes of your problem and chooses the class with highest membership as the final predction.

After calling the `.predict()` function, you can view the fuzzy membership to the test data by calling the `.get_memberships()` function:

`knn.get_memberships()`

This call will print a Python dict with the fuzzy membership for each classes.
