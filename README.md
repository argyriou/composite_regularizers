# Accelerated optimization for composite regularizers

This optimization method solves regularization problems with regularizers of the form *f(x) + h(Bx)* where 
* *f* is a strongly smooth function
* *h* is a nonsmooth function whose proximity operator is easy to compute
* *B* is a linear map. 

The algorithm combines a fixed-point method with Nesterov acceleration. See [Efficient First Order Methods for Linear Composite Regularizers](http://ttic.uchicago.edu/~argyriou/papers/picanest_arxiv.pdf).

