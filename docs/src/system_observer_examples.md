# Using Systems and Observers


## Example 1 - Linear Systems

System types can describe both discrete-time and continuous-time dynamics,
depending on the type of state the system is used with. Calling the `predict`
method for a `LinearSystem` with a `DiscreteState` causes the system to be
treated as discrete and calling `predict` with a `ContinuousState` causes the
system to be continuous. In this example, we will look at both types of
`LinearSystem`.

The code block below demonstrates an implementation of the following
continuous-time model.

$\dot{x}(t) = -x(t)$

We will use a `LinearSystem` with a `ContinuousState` to model these dynamics.
Assuming the initial `ContinuousState` has a value of `2.0` at `t = 0.0`, we
will use the `predict` method to determine the value of the state at time
`t = 2.0`.

```jldoctest
julia> using StateEstimation

julia> state = ContinuousState(2.0)
StateEstimation.ContinuousState{Float64}([2.0], 0.0)

julia> system = LinearSystem(-1.0)
StateEstimation.LinearSystem{Float64}([-1.0], [0.0])

julia> predict(system, state, 2.0)
StateEstimation.ContinuousState{Float64}([0.270671], 2.0)

```

The next example considers a 2-dimensional, discrete-time system. This system
describes a rotation of 0.1 radians about the origin.

$x_k = \left[\begin{array}{cc} \cos(0.1) & -\sin(0.1) \\ \sin(0.1) & \cos(0.1) \end{array} \right] x_{k-1} + w_k$

In the code block below, we build an `UncertainDiscreteState` and a
`LinearSystem`. Notice that the initial state in this case is uncertain, so
in addition to a predicted state vector, we will also get a predicted
covariance.

```jldoctest
julia> using StateEstimation

julia> state = UncertainDiscreteState(ones(2), eye(2))
StateEstimation.UncertainDiscreteState{Float64}([1.0, 1.0], [1.0 0.0; 0.0 1.0], 0)

julia> A = [[cos(0.1), sin(0.1)] [-sin(0.1), cos(0.1)]]
2×2 Array{Float64,2}:
 0.995004   -0.0998334
 0.0998334   0.995004

julia> Q = 0.01*eye(2)
2×2 Array{Float64,2}:
 0.01  0.0
 0.0   0.01

julia> system = LinearSystem(A, Q)
StateEstimation.LinearSystem{Float64}([0.995004 -0.0998334; 0.0998334 0.995004], [0.01 0.0; 0.0 0.01])

julia> predict(system, state, 5)
StateEstimation.UncertainDiscreteState{Float64}([0.398157, 1.35701], [1.05 0.0; 0.0 1.05], 5)

```

The state type input to the `predict` method is always the same as
the output state type. So, predicting an absolute state produces an absolute
state and predicting an uncertain state produces an uncertain result. To predict
an absolute state in the presence of process noise, call `make_uncertain` first
to convert it to its corresponding uncertain format.

The `predict!` method provides a state prediction interface that performs an
in-place update of the provided state.


## Example 2 - Nonlinear Systems

Nonlinear systems allow for more flexible modeling of system dynamics but also
require additional caution in use.

`NonlinearSystem` constructors require a function, F, its Jacobian, dF_dx, and a
covariance describing the process noise. The two `Function` inputs must follow
specific requirements to fit within the `NonlinearSystem` framework.

First, let's look at a discrete case. To define a discrete, nonlinear system,
the user must provide a function with the format `F(k::Integer, x_k::Vector)`.
Even if the state we plan to use with this system is a scalar, the input to this
function will still be a vector. Similarly, the Jacobian must take the form
`dF_dx(k::Integer, x_k::Vector)`. To perform a prediction step, these functions
will be called with the current step count and the current state vector.

Consider the following discrete-time system.

$x_k = x_{k-1}^{1.05}$

The code below demonstrates the use of a `DiscreteState` and a `NonlinearSystem`
to predict the state forward through the model above. This code builds both `F`
and `dF_dx` and creates a `NonlinearSystem` with a process noise covariance of
`0.0`.

```jldoctest
julia> using StateEstimation

julia> state = DiscreteState(10.0)
StateEstimation.DiscreteState{Float64}([10.0], 0)

julia> F(k, x_k) = x_k.^1.05
F (generic function with 1 method)

julia> dF_dx(k, x_k) = 1.05*x_k.^0.05
dF_dx (generic function with 1 method)

julia> system = NonlinearSystem(F, dF_dx, 0.0)
StateEstimation.NonlinearSystem{Float64}(F, dF_dx, [0.0], 0.01, 0.01)

julia> predict(system, state, 10)
StateEstimation.DiscreteState{Float64}([42.5495], 10)

```


## Example 3 - Linear Observers

Unlike the system types, observers take the same form regardless of the type of
state they are used with. In the next code block, we look at a simple observer
of a scalar state with the measurement function

$y = \frac{x}{4} + v$

```jldoctest
julia> using StateEstimation

julia> state = DiscreteState(2.0)
StateEstimation.DiscreteState{Float64}([2.0], 0)

julia> observer = LinearObserver(0.25, 0.01)
StateEstimation.LinearObserver{Float64}([0.25], [0.01])

julia> predict(observer, state)
StateEstimation.DiscreteState{Float64}([0.5], 0)

```

Notice that, even though the measurement model contains noise, the measurement
prediction is not uncertain because the input state was not uncertain. In the
next code sample, we look at propagation of an uncertain state through a
`LinearObserver`.

For this example, consider the following observer. This observer accepts a
2-dimensional input state and produces a scalar measurement of only the first
element.

$y = \left[\begin{array}{cc} 1 & 0 \end{array} \right] x + v$

The code below constructs an `UncertainContinuousState` with some initial
uncertainty, builds the `LinearObserver`, and predicts the state through the
observer. The result is a scalar `UncertainContinuousState`.

```jldoctest
julia> using StateEstimation

julia> state = UncertainContinuousState([1.0, 2.0], 0.1*eye(2))
StateEstimation.UncertainContinuousState{Float64}([1.0, 2.0], [0.1 0.0; 0.0 0.1], 0.0)

julia> H = [1.0 0.0]
1×2 Array{Float64,2}:
 1.0  0.0

julia> R = 0.1
0.1

julia> observer = LinearObserver(H, R)
StateEstimation.LinearObserver{Float64}([1.0 0.0], [0.1])

julia> predict(observer, state)
StateEstimation.UncertainContinuousState{Float64}([1.0], [0.2], 0.0)

```

## Exampled 4 - Nonlinear Observers
