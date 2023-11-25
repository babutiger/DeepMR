## DeepMR is a new neural network validation tool

The "mnist_properties" folder is used to store MNIST data for verification, which can be classified by models awaiting validation.

The "cifar_properties" folder is used to store CIFAR10 data for validation.

The "models" folder stores neural network models for validation.

The "multipath_bp" folder contains the code required to run the aCROWN and MPBP algorithms.

The "result" folder saves experimental results, where the "log" folder contains log data from code execution with additional details, and the "original_result" folder contains raw results obtained from code execution.

The "sources" folder saves MNIST and CIFAR10 datasets downloaded from official sources.

The "utils" folder includes some utility code.

The "verify" folder is used for comparative experiments of verification algorithms, including aCROWN, DeepPoly, DeepSRGR, MPBP, and DeepMR.Running the Python file yields the verification results for the algorithm. We provide a total of 6 neural network models for validation, including 3 MNIST networks and 3 CIFAR10 networks.

It is important to note that we focus on fully connected networks with ReLU activation functions.

In the code for DeepMR and DeepSRGR, the section related to linear programming solvers is parallelized. If your machine has good performance, preferably a multi-core server, you can increase the number of processes. We have set the default number of processes to WORKERS=12 and the number of iterations to 5. Because linear programming solving is slow, more processes will speed up the solving process.

We default to using the CBC solver, but other solvers can also be used.
