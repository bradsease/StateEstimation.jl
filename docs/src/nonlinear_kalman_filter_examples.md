# Nonlinear Filtering


## Example 1 - Extended Kalman Filter with Discrete Dynamics

In this example, we will create an `ExtendedKalmanFilter` to estimate the state
of a scalar nonlinear system described by the following equation.

$x_{k+1} = F(k,x_k) = \sin x_k + w_k$

where $w_k \sim N(0, Q)$. To estimate the state of this nonlinear system, we
also need the Jacobian of $F$. The Jacobian is

$\frac{\partial F(k,x_k)}{\partial x_k} = \cos x_k$

We'll consider the nonlinear observer

$y_k = H(k,x_k) = \cos x_k$

where the Jacobian of $H(k, x_k)$ is

$\frac{\partial H(k,x_k)}{\partial x_k} = -\sin x_k + v_k$

where $v_k \sim N(0, R)$. This example requires us to build a
`NonlinearSystem`, a `NonlinearObserver`, an initial `UncertainContinuousState`,
and a `ExtendedKalmanFilter`. We will also create a simulator to create simulated
measurement data using `make_simulator`.

```jldoctest
using StateEstimation

# Define the system dynamics
F(t,x) = sin.(x)
dF_dx(t,x) = cos.(x)
Q = 0.01
nonlinear_sys = NonlinearSystem(F, dF_dx, Q)

# Define the observer
H(t,x) = cos.(x)
dH_dx(t,x) = -sin.(x)
R = 0.1
nonlinear_obs = NonlinearObserver(H, dH_dx, R)

# Define the initial state
initial_est = UncertainDiscreteState(0.0, 0.1)

# Build the ExtendedKalmanFilter
ekf = ExtendedKalmanFilter(nonlinear_sys, nonlinear_obs, initial_est);

# Create a simulator
simulator = make_simulator(ekf)

# Simulate & process measurements
for idx = 1:10
    true_state, measurement = simulate(simulator, idx)
    process!(ekf, measurement)
end

# output

```

To view the current estimate at any point, access `ekf.estimate`.


## Example 2 - Extended Kalman Filter with Continuous Dynamics

In this example, we will create an `ExtendedKalmanFilter` to estimate the state
of a nonlinear pendulum described by the following dynamics.

$\left[ \begin{array}{c} \dot{x}_1 \\ \dot{x}_2 \end{array} \right] = F(t,\vec{x}) = \left[ \begin{array}{c} x_2 \\ - \sin x_1 \end{array} \right] + w(t)$

where $w(t) \sim N(0, Q)$. To estimate the state of this nonlinear system, we also
need the Jacobian of $F$. The Jacobian is

$\frac{\partial F(t,\vec{x})}{\partial \vec{x}} = \left[ \begin{array}{cc} 0 & 1 \\ -\cos x_1 & 0 \end{array} \right]$

We will use the `LinearObserver` below.

$y(t_k) = x_1(t_k) + v(t_k)$

where $v(t_k) \sim N(0, R)$. This example requires us to build a
`NonlinearSystem`, a `LinearObserver`, an initial `UncertainContinuousState`, and
a `ExtendedKalmanFilter`. We will also create a simulator to create simulated
measurement data using `make_simulator`.

```jldoctest
using StateEstimation

# Define the system dynamics
F(t,x) = [x[2], -sin(x[1])]
dF_dx(t,x) = [0 1; -cos(x[1]) 0]
Q = (0.01*pi/180)*eye(2,2)
nonlinear_sys = NonlinearSystem(F, dF_dx, Q)

# Define the observer
H = [1.0, 0.0]'
R = 0.1*pi/180  # 1 degree variance
linear_obs = LinearObserver(H, R)

# Define the initial state
initial_est = UncertainContinuousState([pi/4, 0.0], diagm([0.1, 0.01])*pi/180)

# Build the ExtendedKalmanFilter
ekf = ExtendedKalmanFilter(nonlinear_sys, linear_obs, initial_est);

# Create a simulator
simulator = make_simulator(ekf)

# Simulate & process measurements
for t = 0.1:0.1:10
    true_state, measurement = simulate(simulator, t)
    process!(ekf, measurement)
end

# output

```

The code above iteratively simulates measurements and processes them with the
`ExtendedKalmanFilter`. To view the current estimate at any point, view
`ekf.estimate`.


## Example 2 - Unscented Kalman Filter


In this example, we will create an `UnscentedKalmanFilter` to estimate the state
of a nonlinear system described by the following equation.

$\vec{x}_{k+1} = F(k,\vec{x}_k) = \left[ \begin{array}{c} \sin(x_{k,2}) \\ - x_{k,1} \end{array} \right] + w_k$

where $w_k \sim N(0, Q)$. To estimate the state of this nonlinear system, we
also need the Jacobian of $F$. The Jacobian is

$\frac{\partial F(k,x_k)}{\partial x_k} = \left[ \begin{array}{cc} 0 & \cos(x_{k,2}) \\ - 1 & 0 \end{array} \right]$

We'll consider the linear observer

$y_k = \vec{x}_k + v_k$

where $v_k \sim N(0, R)$. This example requires us to build a
`NonlinearSystem`, a `LinearObserver`, an initial `UncertainDiscreteState`,
and a `ExtendedKalmanFilter`. We will also create a simulator to create simulated
measurement data using `make_simulator`.

```jldoctest
using StateEstimation

# Define the system dynamics
F(t,x) = [sin(x[2]), -x[1]]
dF_dx(t,x) = [0 cos(x[2]); -1 0]
Q = 0.01*eye(2,2)
nonlinear_sys = NonlinearSystem(F, dF_dx, Q)

# Define the observer
H = eye(2)
R = 0.1*eye(2)
linear_obs = LinearObserver(H, R)

# Define the initial state
initial_est = UncertainDiscreteState([0.0, 2.0], 0.1*eye(2))

# Build the ExtendedKalmanFilter
ukf = UnscentedKalmanFilter(nonlinear_sys, linear_obs, initial_est);

# Create a simulator
simulator = make_simulator(ukf)

# Simulate & process measurements
for idx = 1:10
    true_state, measurement = simulate(simulator, idx)
    process!(ukf, measurement)
end

# output

```
