# Machine Learning experiments
This repository contains Jupyter Notebooks with explanations and implementation of some Machine Learning, Statistics and Data Analysis methods. Each argument is firstly described and theoretically explained and then implemented *without* using any external library.

> Concepts are more important than efficiency.

## Elements of Statistical Learning
I provide my implementation of the methods described in the famous book 
[The Elements of Statistical Learning](https://www.amazon.it/Elements-Statistical-Learning-Inference-Prediction/dp/0387848576).
I try to follow the chapter progression as much as possible. 

Please remember that this code and notebooks are purely for learning purposes and therefore do not constitute a valuable
 source of information and could contain errors.
 
* [(2) Overview of Supervised Learning](./notebooks/Elements_Of_Statistical_Learning/02_Overview_of_Supervised_Learning)
    * [(2.3) Least Squares and Nearest Neighbors](./notebooks/Elements_Of_Statistical_Learning/02_Overview_of_Supervised_Learning/2.3_LeastSquares_and_NearestNeighbors.ipynb)
    * [(2.4) Statistical Decision Theory](./notebooks/Elements_Of_Statistical_Learning/02_Overview_of_Supervised_Learning/2.4_Statistical_Decision_Theory.ipynb)

## Classical Machine Learning
* [Naive Bayes](./notebooks/Naive_Bayes.ipynb )
* [Polynomial Curve fitting](./Polynomial_Curve_Fitting.ipynb): evaluation of models of several degree
* [Linear Regression](./Linear_Regression.ipynb): both frequentist and bayesian approach
* [Linear Classification](./Linear_Classification.ipynb): least squares
* [Principal Component Analysis](./Principal_Component_Analysis.ipynb)
* [Monte Carlo Method](./Monte_Carlo_Method.ipynb)
* [Markov Chains](./Markov_Chains.ipynb)
* [Hidden Markov Model](./Hidden_Markov_Model.ipynb)


## Neural Networks
* [Restricted Boltzmann Machine](./Restricted_Boltzmann_machine.ipynb)

## The code library
In the [src](./src) folder there are some utility functions that I use in the various notebooks, like data generation 
utilities, pre-processing and so on.