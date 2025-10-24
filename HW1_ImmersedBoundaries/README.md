# Numerical PDEs HW #1 - Working with Immersed Boundaries

Daniel Krawciw

## Problem 1

The following is the numerical solution to the Poisson Equation: $$\nabla^2 u = f(x,y).$$ Given dirchelet boundary conditions around the boundaries of $x \in [0, 2\pi]$ and $y \in [0, 2\pi]$ from $u(x,y) = \cos(2x) \sin(2y)$.

!["Heatplot of the numerical solution to the given poisson problem"](output/problem1_1.svg)

!["Error Convergence of the Numerical Method"](output/problem1_2.svg)

Here, we can see that the slopes of both errors match the slope of the reference error. Thus, the error of this method is second error.

## Problem 2

!["Immersed Boundary Plot Solved with Points Displayed"](output/problem2_1.svg)

!["Error Convergence of Solving Immersed Boundary Value Problem"](output/problem2_2.svg)

### part a

I found that the underlying grid stayed the same, at least to the eye. I suspect that this is because the immersed boundary points are not actually forcing a different shape on the underlying "fluid", they are simply matching the values at those points.

We can see here that the solution is the same as the one from problem 1, but with these immersed points that actually carry a charge that is solved for with this problem. Here, we can see the values of the solved charges for our immersed boundary:

![""](output/problem2_1_charges.svg)

## Problem 3

For this problem, we do the same as problem 2, except we fix the boundary with Dirchelet BCs of $0$ and we make the forcing function $f(x,y) = 0$ too.

![""](output/problem3_1_2D.svg)

![""](output/problem3_1_scatter.svg)

## Problem 4

Here, I chose to add "interesting-looking" shapes and observe what the surface looks like.

![""](output/problem4.svg)