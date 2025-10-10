# Numerical PDEs HW #1 - Working with Immersed Boundaries

Daniel Krawciw

## Problem 1

The following is the numerical solution to the Poisson Equation: $$\nabla^2 u = f(x,y).$$ Given dirchelet boundary conditions around the boundaries of $x \in [0, 2\pi]$ and $y \in [0, 2\pi]$ from $u(x,y) = \cos(2x) \sin(2y)$.

!["Heatplot of the numerical solution to the given poisson problem"](output/problem1_1.svg)

!["Error Convergence of the Numerical Method"](output/problem1_2.svg)

Here, we can see that the slopes of both errors match the slope of the reference error. Thus, the error of this method is second error.