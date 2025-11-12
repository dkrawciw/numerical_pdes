# Numerical PDEs Homework #2

Daniel Krawciw

## Problem 1

Using the two-norm, I compared the error convergence of all the methods from this class here:

![Plot comparing the error convergence of all methods](output/compare_all_methods.svg)

As expected, the two methods that only have first order convergence (forward and backward euler) are at the top with the most shallow slopes.

After this, we see that the other methods are have second order accuracy (except for `ODE45`). All of them seem to have this error convergence up to a certain point, then their convergence completely reverses in trend and the method diminishes in convergence. This is visible for RK2, Crank-Nicolson, and BDF2. This is most likely due to time error relative to spacial error (more about this later).

Finally, `ODE45` just looks like a constant line across the plot. This is because it is adaptive and you don't just set the number of points, the method itself defines the necessary points based off of error that it calculates between the solutions of RK4 and RK5. So I left it as a constant line here since I think it illustrates how `ODE45` works.

<!-- ### Forward Euler

![Forward Euler Error Convergence](output/problem1_ForwardEuler.svg)

### Backward Euler

![Backward Euler Error Convergence](output/problem1_BackwardEuler.svg)

### Crank-Nicolson

![Crank-Nicolson Error Convergence](output/problem1_CrankNicolson.svg)

### Explicit Midpoint (RK2)

![Runge Kutta 2 (Explicit Midpoint) Error Convergence](output/problem1_RK2.svg)

### BDF2

Here, I used RK2 first to get two points, then I was able to use BDF2 with a history point.

![Backward Difference 2 Error Convergence Plot](output/problem1_BDF2.svg)

### ODE45

![ODE45 Error Convergence Plot](output/problem1_ODE45.svg) -->

## Problem 2
