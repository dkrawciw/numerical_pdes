# Numerical PDEs Homework #0

Daniel Krawciw

## Problem 1

The task here is to numerically calculate the first derivative of the function $$f(x) = \exp \left(\sin t \right)$$ over the interval $0 \leq x \leq 2 \pi$. The method here is to use a type one grid and use a centered difference formula over the inner points of the grid, and use a one-sided difference method on the edge points.

The error plot displays a strange difference in order between the $2$-norm and $\infty$-norm. The $\infty$-norm gives an error order of about $1$ whereas the $2$-norm is steeper. The difference in order is due to using 1st order approximations at the edges of the grid. Because the $\infty$-norm is calculated with the maximum error component, the edges stand out compared to the $2$-norm.

![](output/Problem1_1.svg)

![](output/Problem1_2.svg)

## Problem 2

Here, we have a similar situation to problem 1, but we are computing the second derivative with periodic boundary conditions. Something I noticed here, which was different than the first problem, was I had to discard the endpoint of the discrete grid because it's technically the same point as the first point. Having it caused issues with the derivative at endpoints.

For the error, we see that the error orders are roughly the same. This confirms the suspicions from problem 1 where their errors were noticeably different as that method combined different order derivative methods whereas, here, the same method is used on the whole grid.

![](output/Problem2_1.svg)

![](output/Problem2_2.svg)

## Problem 3

Now, we must solve the ODE $$\frac{d^2 u}{dx^2} + \sin x \frac{du}{dx} + u(x) = f(x)$$ over $0 \leq x \leq 5$. Given that the solution would be $u(x) = \sin x$, I found that the forcing function $f(x) = \sin x \cos x$.

Initially, the error plot looked a bit strange where it had bends seemingly out of no where. While exploring, I found that the order of convergence had less of those the more grid points I used to compute the solution. I settled on going from about 10 ponts to 10,000 points.

![](output/Problem3_1.svg)

![](output/Problem3_2.svg)

## Problem 4

These plots are given by setting the boundary of the given box $$ 0 \leq x \leq 5, 0 \leq y \leq 5$$ with values from $\sin x \cos y$, and then using second order difference approximations to get the solution over the rest of the domain.

![](output/Problem4_1.svg)

![](output/Problem4_2.svg)

If we apply Neumann conditions on all boundaries, we get the following result:

![](output/Problem4_3.svg)

This is clearly not the same figure as the solution given Dirichlet boundary conditions. There a couple reasons for this, the first of which is that, when given Neumann boundary conditions, we do not have a unique solution, so there are infinitely many valid solutions because there are more unknowns than equations. The second is related to the first. It's that we have a starting "velocity" of the wave, but not a starting position of the wave.