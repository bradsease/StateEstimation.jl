# Batch Estimation


## Example 1 - Linear Least Squares with Discrete Dynamics

In this example, we will create a linear LeastSquaresEstimator to estimate the
state of the system

$x_k = \left[\begin{array}{cc} \cos(0.1) & -\sin(0.1) \\ \sin(0.1) & \cos(0.1) \end{array} \right] x_{k-1}$

with the observer

$y_k = \left[\begin{array}{cc} 1 & 0 \\ 0 & 1 \end{array} \right] x_k + v_k$

where $w_k \sim N(0, Q)$ and $v_k \sim N(0, R)$. This example requires us to
build a `LinearSystem`, a `LinearObserver`, an initial `UncertainDiscreteState`,
and a `LeastSquaresEstimator`. We will also create a simulator to create
simulated measurement data using `make_simulator`.

The code for this example is shown below.

```jldoctest
using StateEstimation

# Define the system dynamics
A = [[cos(0.1), sin(0.1)] [-sin(0.1), cos(0.1)]]
linear_sys = LinearSystem(A)

# Define the observer
H = eye(2)
R = 0.1*eye(2)
linear_obs = LinearObserver(H, R)

# Define the initial state
initial_est = UncertainDiscreteState([1.0, 1.0], 0.5*eye(2))

# Build the LeastSquaresEstimator
bls = LeastSquaresEstimator(linear_sys, linear_obs, initial_est);

# Create a simulator
simulator = make_simulator(bls)

# Simulate & process measurements
for i = 0:10
    true_state, measurement = simulate(simulator, i)
    add!(bls, measurement)
end

# Update the internal estimate
solve!(bls)

# output

```

The code above iteratively simulates measurements and adds them to the
`LeastSquaresEstimator`. To compute a new estimated state, use `solve` or
`solve!`.



## Example 2 - Linear Least Squares with Continuous Dynamics


## Example 3 - Nonlinear Least Squares
