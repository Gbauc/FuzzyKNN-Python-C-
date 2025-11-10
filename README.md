An implementation of FuzzyKNN in C++ with Python integration, based on the reference [A fuzzy K-nearest neighbors](https://ieeexplore.ieee.org/document/6313426).

You need to be aware that this implementation will only works in Linux systems, the conversion of C++ code to a Python Module was compiled using g++ in a Linux system, generating a `.so` file. To run this application in Windows system, you to recompile the source code in a Windows enviroment using an C++ compiler of our preference and have pybind11 installed in your machine, to generate the Python Module.

First, get the `.so` file and `.py` file in the same folder. Then, you can start using the module by importing him:

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

The `.fit()` function will store the training data in memory and compute the fuzzy membership to each class of your classification problem to each training point. While the `.predict()` fuction will compute the fuzzy membership of the test data to each classes of your problem and chose the class with highest membership as the final predction.

After calling the `.predict()` function, you can see the fuzzy membership to the test data by calling the `.get_memberships()` function:

`knn.get_memberships()`

This call will print a Python dict with the fuzzy membership for each classes.
