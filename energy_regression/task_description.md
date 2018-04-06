# Sample-Weighted Kernel Ridge Regression for Neutrino Energy Reconstruction
## Motivation
In the past few years, we have seen the application of state-of-the art machine learning techniques, primarily the convolutional neural network (CNN), to do particle identification (PID) in our experiment. What physicists call PID belongs to the classification task in machine learning. We have seen tremendous improvements in all metrics with this technique over the former more traditional ways, such as a log-likelihood based method.

Equally important is our events' energy reconstruction, corresponding to machine learning's regression task. Curiously enough, we are still using the relatively simpler techniques to model our nonlinear energy response. In particular, we use the spline fit, basically our home-brew [MARS](https://en.wikipedia.org/wiki/Multivariate_adaptive_regression_splines) model. This is not saying we have no intention to apply the more advanced regression techniques to this problem. Actually, we do have a group of colleagues applying the same network structure the PID uses, but adding one regression layer, to reconstruct the energy. It is still under intense investigation at the time of writing.

Months ago I happened to grab a [machine learning textbook](https://www.amazon.com/Machine-Learning-Optimization-Perspective-Developers/dp/0128015225/ref=sr_1_36?ie=UTF8&qid=1522357862&sr=8-36&keywords=machine+learning), and found a class of techniques called kernel methods to have very beautiful mathematical structure. I became fascinated with those techniques, and started to have ideas to try them out on our data and see if that would improve our energy estimation.

The answer is **yes**. Now, let me walk you through.
## Ridge Regression
Let's start with the most basic. Given $N$ training samples $(\mathbf{x}_i,y_i)$, where $\mathbf{x}_i$'s $\in\mathbb{R}^\ell$ are the regressors and $y_i$'s $\in\mathbb{R}$ are the targets, we want to find a linear function in $\mathbf{w}$, $$f_{\mathbf{w}}(\mathbf{x})=\mathbf{w}^T\mathbf{x}$$, that minimizes the square error cost function $$C(\mathbf{w})=\frac{1}{2}\sum_{i=1}^{N}(y_i-\mathbf{w}^T\mathbf{x}_i)^2$$
By differentiation with respect to $\mathbf{w}$, it is equivalent to solving the **normal equation**,
$$\mathbf{X}^T\mathbf{X}\mathbf{w}=\mathbf{X}^T\mathbf{y}$$
, where $\mathbf{X}$ is the so called _design matrix_, $$\mathbf{X}=\begin{pmatrix}\mathbf{x}_1^T \\ \vdots \\ \mathbf{x}_N^T \end{pmatrix}$$
If $\mathbf{X}^T\mathbf{X}$ is invertible, $\mathbf{w}$ is readily solved. However, in practice, the predictor variables are very often nearly linearly dependent, leading to a phenomenon called [multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity). In such cases, the resulting $\mathbf{w}$ becomes highly sensitive to variations in training samples, leading to overfitting. One of the most popular remedies is the [Tikhonov regularization](https://en.wikipedia.org/wiki/Tikhonov_regularization): Instead of minimizing the above cost function, minimize this one $$C(\mathbf{w})=\frac{1}{2}\sum_{i=1}^{N}(y_i-\mathbf{w}^T\mathbf{x}_i)^2+\frac{1}{2}\alpha\Vert\mathbf{w}\Vert^2$$, resulting in $\mathbf{w}=(\mathbf{X}^T\mathbf{X}+\alpha\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$. In regression jargon, this is called _ridge regression_.
## Dual Form
$\mathbf{w}$ can be rewritten as follows:
$$
  \mathbf{X}^T(\mathbf{X}\mathbf{X}^T+\alpha\mathbf{I})=(\mathbf{X}^T\mathbf{X}+\alpha\mathbf{I})\mathbf{X}^T \\
  \Rightarrow \mathbf{X}^T(\mathbf{X}\mathbf{X}^T+\alpha\mathbf{I})^{-1}=(\mathbf{X}^T\mathbf{X}+\alpha\mathbf{I})^{-1}\mathbf{X}^T \\
  \Rightarrow \mathbf{w}=(\mathbf{X}^T\mathbf{X}+\alpha\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}=\mathbf{X}^T(\mathbf{X}\mathbf{X}^T+\alpha\mathbf{I})^{-1}\mathbf{y}
$$
Given a test sample $\hat{\mathbf{x}}$, its predicted value is $\hat{y}=\hat{\mathbf{x}}^T\mathbf{w}=\hat{\mathbf{x}}^T\mathbf{X}^T(\mathbf{X}\mathbf{X}^T+\alpha\mathbf{I})^{-1}\mathbf{y}=\sum_{i=1}^{N}a_i\hat{\mathbf{x}}^T\mathbf{x}_i$.  
Here,
$$
\mathbf{a}=(\mathbf{X}\mathbf{X}^T+\alpha\mathbf{I})^{-1}\mathbf{y}=(\mathbf{G}+\alpha\mathbf{I})^{-1}\mathbf{y}
$$
, where $\mathbf{G}$ is the [Gram matrix](https://en.wikipedia.org/wiki/Gramian_matrix), $G_{ij}=\mathbf{x}_i^T\mathbf{x}_j$.  
This is the **dual form** of the regression problem, where every appearance of the regressor variable is in the form of an inner product with another regressor.
## Feature Map
The first idea to form a nonlinear model (in $x$) is using polynomials. For example, to fit a second order polynomial, we form the regression function like this:
$$
f_\mathbf{w}(x)=w_0+w_1x+w_2x^2=\mathbf{w}^T\mathbf{\phi}(x)
$$
Here, $\mathbf{\phi}:\mathbb{R}\rightarrow\mathbb{R}^3$, $\mathbf{\phi}(x)=(1,x,x^2)^T$, is called a feature map. Instead of doing nonlinear regression in the input space $\mathbb{R}$, we can do linear regression in the higher dimentional feature space $\mathbb{R}^3$, and all the formula previously mentioned apply, as long as we replace $x$ with $\mathbf{\phi}(x)$. (I have been sloppy in the constant term. However, they are easy to deal with and does not change the formula.)

To summarize, to come up with a nonliear model, we engineer a nonlinear map to a higher dimentional Hilbert space, hoping for picking up the nonlinear features by the feature map, and do linear regression in the feature space.
## Kernel Trick
As mentioned before, in dual form, every occurrence of a regressor comes in the form of an inner product with another regressor. Therefore, in doing regression in the feature space, we only care about **one** real number $\mathbf{\phi}(\mathbf{x})^T\mathbf{\phi}(\mathbf{x'})$, not $M$ real numbers of the actual image $\mathbf{\phi}(\mathbf{x})$, where $M$ is the dimensionality of the range of $\mathbf{\phi}$.  
That being said, if we can find a kernel function $k:\mathbb{R}^\ell\times\mathbb{R}^\ell\rightarrow\mathbb{R}$ such that $k(\mathbf{x},\mathbf{x}')=\mathbf{\phi}(\mathbf{x})^T\mathbf{\phi}(\mathbf{x}')$, $\forall \mathbf{x}, \mathbf{x}' \in \mathbb{R}^\ell$, we can solve the high-dimensional linear regression problem in the input space without casting to the computationally heavy, sometimes even impossible in the infinite dimensional case, feature space!

## Reproducing Kernel Hilbert Space (RKHS)
Questions surface. First of all, does there even exist a feature map with an associated Hilbert space such that the inner product of each pair of the images of input variables can be represented by a kernel function?

Here comes the beauty of functional analysis.

The [Moore-Aronszajn theorem](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space#Moore–Aronszajn_theorem) states that every symmetric, positive definite kernel defines a unique Hilber space, the _reproducing kernel Hilbert space_ (RKHS).  
Let me highlight the definitions here. A kernel function is
* symmetric if $k(\mathbf{x},\mathbf{x}')=k(\mathbf{x}',\mathbf{x})$
* positive definite if $\sum_{i,j=1}^{n}c_ic_jk(\mathbf{x}_i,\mathbf{x_j})\ge0$, for any $n\in\mathbb{N}$, $\mathbf{x}_1,...,\mathbf{x}_n\in\mathbb{R}^\ell$, and $c_1,...,c_n\in\mathbb{R}$

Most of the popular kernel functions satisty these conditions (with the exception of the sigmoid kernel maybe, depending on how you define "popular"), therefore we will not worry about the details here.
### Examples of Kernel Functions
Here I list some examples of kernel functions.
* The Gaussian kernel, where $\gamma$ is a parameter. $$k(\mathbf{x},\mathbf{y})=exp\left(-\gamma\Vert\mathbf{x}-\mathbf{y}\Vert^2\right)$$
* The polynomial kernel of order $r$, where $r$ is a parameter. $$k(\mathbf{x},\mathbf{y})=(\mathbf{x}^T\mathbf{y})^r$$
* The polynomial kernel of order up to $r$, where $r$ is a parameter. $$k(\mathbf{x},\mathbf{y})=(\mathbf{x}^T\mathbf{y}+1)^r$$

### An Example of Reconstructing the Feature Map from a Kernel
Since Gaussian kernel has many attractive features, I will stick to just this kernel from now on. The page 11 of [this slides](https://www.csie.ntu.edu.tw/~cjlin/talks/kuleuven_svm.pdf) details how we can actually recover the feature map from the Gaussian kernel. In the 1D input case, the feature map of the Gaussian kernel is
$$\mathbf{\phi}(x)=e^{-\gamma x^2}\left(1,\sqrt{\frac{2\gamma}{1!}}x,\sqrt{\frac{(2\gamma)^2}{2!}}x^2,...\right)^T$$
Therefore, a Gaussian kernel is very much like a local (controlled by $e^{-\gamma x^2}$) polynomial map of all order! This is probably why the Gaussian kernel is the default one to try out and works so well in many situations.

Now, if we replace every inner product in the solution to the linear regression function formula, we roughly rediscover the **representer theorem**.

### Representer Theorem
Fix a kernel $k$, 