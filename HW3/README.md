# Homework #3 - Allen Cahn

Daniel Krawciw

## Question 1

![](output/problem1.svg)

![](output/problem1_mass_plot.svg)

## Question 2

### Part a

We will show that, given $\beta(t)$, our PDE is conservative:

$$
\frac{\partial}{\partial t} \int_\Omega \phi dA = 0
$$

Start by taking the PDE and integrating both sides by the area $\Omega$.

$$
\begin{align*}
    \int_\Omega \phi_t dA &= \int_\Omega \left[ \epsilon^2 \nabla^2\phi - F^\prime(\phi) + \beta(t) \sqrt{F(\phi)} \right]dA\\
    \frac{\partial}{\partial t} \int_\Omega \phi dA &= \epsilon^2 \int_\Omega \nabla^2\phi dA - \int_\Omega F^\prime(\phi) dA + \beta(t) \int_\Omega \sqrt{F(\phi)}dA\\
    &= \epsilon^2 \int_\Omega \nabla^2\phi dA - \int_\Omega F^\prime(\phi) dA + \int_\Omega F^\prime(\phi)dA\\
    &= \epsilon^2 \int_\Omega \nabla^2\phi dA\\
    &= \oint_{\partial \Omega} \nabla \cdot \phi \cdot n dS = \oint_{\partial \Omega} (0) \cdot n dS\\
    &= 0
\end{align*}
$$

We knew that $\nabla \phi = 0$ because of our given neumann bcs.

### Part b

$$
\begin{align*}
    \frac{\phi^{n+1} - \phi^{**}}{\Delta t} &= \beta^{n+1} \sqrt{F(\phi^{**})}\\
    \sum_{i,j} \frac{\phi^{n+1} - \phi^{**}}{\Delta t} &= \sum_{i,j} \beta^{n+1} \sqrt{F(\phi^{**})}\\
    \frac{1}{\Delta t}  \sum_{i,j} \left[\phi^{n+1} - \phi^{**} \right] &= \beta^{n+1} \sum_{i,j} \sqrt{F(\phi^{**})}\\
    \beta^{n+1} &= \frac{1}{\Delta t} \frac{\sum_{i,j} \left[\phi^{n+1} - \phi^{**} \right]}{\sum_{i,j} \sqrt{F(\phi^{**})}}\\
\end{align*}
$$

### Part c

![](output/problem2.svg)

![](output/problem2_mass_plot.svg)

For this problem, I wasn't able to make the mass *exact* but I was able to reduce the mass drift by a lot. I suspect it has to do with a dropped variable or something like that that I could not find in my code.

## Question 3

I decided to play around with initial conditions and see what would happen after the 50 second time range with Allen Cahn.

![](output/problem3_random_ic.svg)

![](output/problem3_waves_ic.svg)

![](output/problem3_inf_symbol.svg)

![](output/problem3_infty_with_noise.svg)

I think what is surprising to me is the missing values in some of the plots. If these were just zero values, they would look black in the solution plot, so I believe that they are NaN values, and I'm not sure how they got there. I also see it with really extreme initial conditions. For example, the waves initial condition, we have values at 1 and -1 and those are the areas that eventually disappear.

I am not sure why the infinity solution looks the way it does, but I did think that it was interesting that adding noise throughout the initial condition that was much less significant in magnitude to the values in the infinity still made the solution more mellow.