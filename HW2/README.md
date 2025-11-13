# Numerical PDEs Homework #2

Daniel Krawciw

## Problem 1

The following are figures that look at the error convergence of different time-stepping methods. I am only using the two-norm for the following error calculations.

![Figure comparing the error convergence of first order methods](output/compare_first_order_methods.svg)

![Figure comparing the error convergence of second order methods](output/compare_second_order_methods.svg)

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

| Time Stepper      | 5e-2 | 1e-3 | 5e-6 |
|-------------------|------|------|------|
| Forward Euler     |   2506   |   5438   |   too many to compute   |
| Backward Euler    |   1105   |   6512   |   too many to compute   |
| Crank-Nicolson    |   215   |   1495   |   7196   |
| RK2               |   2507 (couldn't have fewer than this)   |   2507   |   7184   |
| BDF2              |   253   |   1750   |   7196   |

### Calculating cost for each method

Forward Euler Cost = (num_of_steps) * (1 matrix multiplication * (5*64^2 cost per matrix mult.)) = (num_of_steps) * 20480

Backward Euler Cost = (cholesky factorization cost) + (num_of_steps) * (1 sparse matrix solve * (64^2 * log(64^2) cost per matrix solve)) = 262144 + (num_of_steps) * 14796.23

Crank-Nicolson Cost = (cholesky factorization cost) + (num_of_steps) * ( (1 sparse matrix solve * 14796.23) + (1 sparse matrix mult. * 20480) ) = 262144 + (num_of_steps) * 35276.23

RK2 Cost = (num_of_steps) * (2 sparse matrix solves * 64^2 * log(64^2) cost per matrix solve ) = (num_of_steps) * 29592.45

BDF2 Cost = (cholesky factorization cost) + (cost of one RK2 iteration) + (num_of_steps - 1) * 14796.22 cost per iteration of BDF2 = 262144 + 29592.45 + (num_of_steps - 1) * 14796.22

### Cost Table

| Time Stepper      | 5e-2 | 1e-3 | 5e-6 |
|-------------------|------|------|------|
| Forward Euler     |   51322880   |   111370240   |   --   |
| Backward Euler    |   16349830.11   |   96353049.76   |   --   |
| Crank-Nicolson    |   7846533.45   |   53000107.85   |   254109895.08   |
| RK2               |   74188272.15   |   74188272.15   |   212592160.8   |
| BDF2              |   4020383.89   |   26170325.23   |   106750539.35   |

I found that, for each error tolerance, BDF2 won in efficiency. This makes sense as it's an implicit method that relies of history points ($u^{n-1}$) so it's stable. Also, there is just one solve and one cholesky factorization, so there is much simplicity in the calculation itself.

I thought that maybe RK2 would be a good method, but I found that because the method was not as stable as an implicit method, that "competitive step sizes" (step sizes small enough to bring the cost down) would blow up.
