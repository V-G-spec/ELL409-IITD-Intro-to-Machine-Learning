# ELL409 - Machine Intelligence and Learning

## About
Assignments done as part of the course ELL409 taught at IIT Delhi (2021 Fall Semester). Each of the assignments contain flexible, user-driven ML models built from scratch 
with extensive and exhaustive hyperparameter tuning, visualizations and analysis.

You can find the detailed description of each assignment in their individual folders. A high level overview of them is as follows:
### Assignment 1
- The objective here is to implement the concepts of regression via polynomial curve fitting. Implemented using 2 methods: Moore-Penrose Pseudoinverse and Gradient Descent.
- Linear regression on time-series dataset.
- [Assignment Report](https://github.com/V-G-spec/ELL409-IITD-Machine-Intelligence-Learning-/blob/master/Assignment%201/2019EE10143/2019EE10143_Report.pdf)

### Assignment 2
- Here, the objective is to experiment with the use of SVMs for both binary and multiclass classification problems by comparing accuracy, speed and support vectors by 
2 approaches: LIBSVM (sklearn.svm.svc) and CVX (Derivations inside the report)
- Implement the [simplified SMO algorithm](https://web.iitd.ac.in/~sumeet/smo.pdf) (part of [CS229](https://cs229.stanford.edu/) course materials at Stanford), 
and train a few SVMs using your implementation.
- Implement [Full SMO Algorithm](https://web.iitd.ac.in/~sumeet/tr-98-14.pdf) by building on the simplified version as above. Explain in detail how you chose 
which Lagrange multipliers to optimise.
- All parts have been programmed to work for 4 kernels: Linear, Polynomial, RBF (Gaussian), Sigmoid.
- [Assignment Report](https://github.com/V-G-spec/ELL409-IITD-Machine-Intelligence-Learning-/blob/master/Assignment%202/2019EE10143/Report.pdf)

### Assignment 3
- Objective is to experiment with the use of Neural Networks for a multiclass classification problem, and try and interpret the
high-level or hidden representations learnt by it. 
- Implement a highly flexible neural net and compare results with a neural network library (Keras).
- To try and understand the effects of various parameter choices such as the number of hidden layers, the number of hidden neurons, and the learning rate.
- [Assignment Report](https://github.com/V-G-spec/ELL409-IITD-Machine-Intelligence-Learning-/blob/master/Assignment%203/2019EE10143/2019EE10143_Report.pdf)

## Author
[Vansh Gupta](https://github.com/V-G-spec)  
Undergraduate student, Electrical Engineering Department  
Indian Institute of Technology, Delhi

## License
Copyright -2021 -Indian Institute of Technology, Delhi

Part of course ELL409: Machine Intelligence and Learning (Taught by professors 
[Sumeet Agarwal](https://web.iitd.ac.in/~sumeet/) and [Jayadeva](https://web.iitd.ac.in/~jayadeva/))

## References
[1] John Platt. Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines. Tech. rep. Microsoft Research, 1998.
