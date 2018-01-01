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

$x_k = \left[\begin{array}{cc} \cos(0.1) & -\sin(0.1) \\ \sin(0.1) & \cos(0.1) \end{array} \right] x_{k-1} + w_k$

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

The `predict!` method also provides an in-place update of the provided state.


## Example 2 - Nonlinear Systems


## Example 3 - Linear Observers

Unlike the system types, observers take the same form regardless of the type of
state they are used with.

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

$y = \left[\begin{array}{cc} 1 & 0 \end{array} \right] x + v$

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
