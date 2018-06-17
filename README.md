# Accelerated optimization for composite regularizers

This optimization method minimizes functions of the form *f(x) + h(Bx)* where 
* *f* is a strongly smooth function
* *h* is a nonsmooth function whose proximity operator is easy to compute
* *B* is a linear map.

An example of such optimization problems is regularization with penalties such as the composition of the L<sub>1</sub> norm or Group Lasso with a linear map. 

The algorithm combines a fixed-point method with Nesterov acceleration. See [Efficient First Order Methods for Linear Composite Regularizers](http://ttic.uchicago.edu/~argyriou/papers/picanest_arxiv.pdf).

