# Batch Estimation


## Example 1 - Linear Least Squares with Discrete Dynamics

In this example, we will create a linear LeastSquaresEstimator to estimate the
state of the system

$x_k = \left[\begin{array}{cc} \cos(0.1) & -\sin(0.1) \\ \sin(0.1) & \cos(0.1) \end{array} \right] x_{k-1}$

with the observer

$y_k = \left[\begin{array}{cc} 1 & 0 \\ 0 & 1 \end{array} \right] x_k + v_k$

where $v_k \sim N(0, R)$. This example requires us to build a `LinearSystem`,
a `LinearObserver`, an initial `UncertainDiscreteState`, and a
`LeastSquaresEstimator`. We will also create a simulator to create simulated
measurement data using `make_simulator`.

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

In this example, we will create a linear LeastSquaresEstimator to estimate the
state of the system

$\dot{x}(t_k) = -0.1 x(t_k)$

with the observer

$y(t_k) = x(t_k) + v_k$

where $v_k \sim N(0, R)$. This example requires us to build a `LinearSystem`, a `LinearObserver`, an initial `UncertainDiscreteState`, and a
`LeastSquaresEstimator`. We will also create a simulator to create simulated measurement data using `make_simulator`.

```jldoctest
using StateEstimation

# Define the system dynamics
A = -0.1
linear_sys = LinearSystem(A)

# Define the observer
H = 1.0
R = 0.1
linear_obs = LinearObserver(H, R)

# Define the initial state
initial_est = UncertainContinuousState(10.5, 0.5)

# Build the LeastSquaresEstimator
bls = LeastSquaresEstimator(linear_sys, linear_obs, initial_est);

# Create a simulator
simulator = make_simulator(bls)

# Simulate & process measurements
for t = 0:0.1:10
    true_state, measurement = simulate(simulator, t)
    add!(bls, measurement)
end

# Update the internal estimate
solve!(bls)

# output

```


## Example 3 - Nonlinear Least Squares

In this example, we will create a `NonlinearLeastSquaresEstimator` to estimate
the state of a nonlinear pendulum described by the following dynamics.

$\left[ \begin{array}{c} \dot{x}_1 \\ \dot{x}_2 \end{array} \right] = F(t,\vec{x}) = \left[ \begin{array}{c} x_2 \\ - \sin x_1 \end{array} \right]$

To estimate the state of this nonlinear system, we also need the Jacobian of
$F$. The Jacobian is

$\frac{\partial F(t,\vec{x})}{\partial \vec{x}} = \left[ \begin{array}{cc} 0 & 1 \\ -\cos x_1 & 0 \end{array} \right]$

We will use the `LinearObserver` below.

$y(t_k) = x_1(t_k) + v_k$

where $v_k \sim N(0, R)$. This example requires us to build a `NonlinearSystem`,
a `LinearObserver`, an initial `UncertainContinuousState`, and a
`LeastSquaresEstimator`. We will also create a simulator to create simulated
measurement data using `make_simulator`.

```jldoctest
using StateEstimation

# Define the system dynamics
F(t,x) = [x[2], -sin(x[1])]
dF_dx(t,x) = [0 1; -cos(x[1]) 0]
Q = zeros(2,2)
nonlinear_sys = NonlinearSystem(F, dF_dx, Q)

# Define the observer
H = [1.0, 0.0]'
R = 0.1*pi/180  # 1 degree variance
linear_obs = LinearObserver(H, R)

# Define the initial state
initial_est = UncertainContinuousState([pi/4, 0.0], diagm([0.1, 0.01])*pi/180)

# Build the NonlinearLeastSquaresEstimator
bls = NonlinearLeastSquaresEstimator(nonlinear_sys, linear_obs, initial_est);

# Create a simulator
simulator = make_simulator(bls)

# Simulate & process measurements
for t = 0.1:0.1:10
    true_state, measurement = simulate(simulator, t)
    add!(bls, measurement)
end

# Update the internal estimate
solve!(bls)

# output

```

We can view the result with

```
julia> bls.estimate.x
2-element Array{Float64,1}:
 0.789735
 0.0138364

 julia> bls.estimate.P
 2×2 Array{Float64,2}:
   0.000175727  -6.3447e-5  
  -6.3447e-5     0.000175414
```

Note that results will differ from run to run due to random noise in the
initial state and measurements. To compare against the truth, store or display
`simulator.true_state` before the main loop.
