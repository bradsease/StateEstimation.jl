# Kalman Filtering

## Example 1 - Scalar Discrete Linear Kalman Filter

In this example, we will create a KalmanFilter to estimate the state of the
scalar system

$\dot{x}(t_k) = 1.05 x(t_k)$

with the observer

$y(t_k) = 1.0 x(t_k) + v(t_k)$

with $v(t_k) \sim N(0, 0.01)$. This example requires us to build a `LinearSystem`,
a `LinearObserver`, an initial `UncertainDiscreteState`, and a `KalmanFilter`.
We will also create a simulator to create simulated measurement data using
`make_simulator`.

The code for this example is shown below.

```jldoctest
using StateEstimation

# Define the system dynamics
linear_sys = LinearSystem(1.05)

# Define the observer
linear_obs = LinearObserver(1.0, 0.01)

# Define the initial state
initial_est = UncertainDiscreteState(0.25, 0.1)

# Build the KalmanFilter
kf = KalmanFilter(linear_sys, linear_obs, initial_est);

# Create a simulator
simulator = make_simulator(kf)

# Simulate & process measurements
for i = 1:10
    true_state, measurement = simulate(simulator, i)
    process!(kf, measurement)
end

# output

```


## Example 2 - Discrete Linear Kalman Filter

In this example, we will create a KalmanFilter to estimate the state of the
system

$x_k = \left[\begin{array}{cc} \cos(0.1) & -\sin(0.1) \\ \sin(0.1) & \cos(0.1) \end{array} \right] x_{k-1} + w_k$

with the observer

$y_k = \left[\begin{array}{cc} 1 & 0 \\ 0 & 1 \end{array} \right] x_k + v_k$

where $w_k \sim N(0, Q)$ and $v_k \sim N(0, R)$. This example requires us to
build a `LinearSystem`, a `LinearObserver`, an initial `UncertainDiscreteState`,
and a `KalmanFilter`. We will also create a simulator to create simulated
measurement data using `make_simulator`.

The code for this example is shown below.

```jldoctest
using StateEstimation

# Define the system dynamics
A = [[cos(0.1), sin(0.1)] [-sin(0.1), cos(0.1)]]
Q = 0.01*eye(2)
linear_sys = LinearSystem(A, Q)

# Define the observer
H = eye(2)
R = 0.1*eye(2)
linear_obs = LinearObserver(H, R)

# Define the initial state
initial_est = UncertainDiscreteState([1.0, 1.0], 0.1*eye(2))

# Build the KalmanFilter
kf = KalmanFilter(linear_sys, linear_obs, initial_est);

# Create a simulator
simulator = make_simulator(kf)

# Simulate & process measurements
for i = 1:10
    true_state, measurement = simulate(simulator, i)
    process!(kf, measurement)
end

# output

```

## Example 3 - Continuous Linear Kalman Filter

In this example, we will create a KalmanFilter to estimate the state of the
system

$\dot{x}(t_k) = \left[\begin{array}{cc} 0 & 1 \\ -1 & 0 \end{array} \right] x(t_k)$

with the observer

$y(t_k) = \left[\begin{array}{cc} 1 & 0 \\ 0 & 0 \end{array} \right] x(t_k) + v(t_k)$

with $v(t_k) \sim N(0, R)$. This example requires us to build a `LinearSystem`,
a `LinearObserver`, an initial `UncertainContinuousState`, and a `KalmanFilter`.
We will also create a simulator to create simulated measurement data using
`make_simulator`.

The code for this example is shown below.

```jldoctest
using StateEstimation

# Define the system dynamics
A = [[0.0, 1.0] [-1.0, 0.0]]
linear_sys = LinearSystem(A)

# Define the observer
H = [1.0 0.0]
R = 0.1
linear_obs = LinearObserver(H, R)

# Define the initial state
initial_est = UncertainContinuousState([1.0, 2.0], 0.1*eye(2))

# Build the KalmanFilter
kf = KalmanFilter(linear_sys, linear_obs, initial_est);

# Create a simulator
simulator = make_simulator(kf)

# Simulate & process measurements
for i = 1:10
    true_state, measurement = simulate(simulator, kf.estimate.t+0.1)
    process!(kf, measurement)
end

# output

```
