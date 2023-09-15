## DeepMR is a new neural network validation tool

The "mnsit_properties" folder is used to store MNIST data to be validated, which can all be prepared for classification by the models awaiting validation.

The "models" folder stores the neural network models to be validated.

The "multipath_bp" folder contains the code necessary to run the aCROWN and MPBP algorithms.

The "result" folder holds the experimental results.

The "utils" folder contains some utility code.

The "verify" folder is for comparative experiments of validation algorithms, including aCROWN, DeepPoly, DeepSRGR, MPBP, and DeepMR. The MPBP algorithm is implemented within the DeepMR algorithm. To obtain MPBP results, run the "mnist_robustness_radius" function in the "deepmr_mnist_new_10x80.py" file. To obtain DeepMR results, run the "mnist_robustness_radius_lp" function. Running other Python files will provide validation results for that specific algorithm.

We have only provided results for a 10x80 MNIST neural network as an example, but further validation with different types of datasets and post-training validation of neural networks, such as the CIFAR dataset, is possible.

It's important to note that we are focusing on fully connected networks with ReLU activation functions.

In the code for DeepMR and DeepSRGR, the section related to linear programming solvers is parallelized. If your machine has good performance, preferably a multi-core server, you can increase the number of processes. We have set the default number of processes to WORKERS=12 and the number of iterations to 5. Because linear programming solving is slow, more processes will speed up the solving process.

We default to using the CBC solver, but other solvers can also be used.