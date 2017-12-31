# StateEstimation Documentation


## Example 1 - Creation

The StateEstimation module contains four basic types of states: `DiscreteState`,
`ContinuousState`, `UncertainDiscreteState`, and `UncertainContinuousState`.

A `DiscreteState` contains a state vector, `x`, and an integer-valued time
step, `t`. When creating any type of state, the time value may be neglected to
indicate a default of zero. The example below shows two ways of creating a
scalar `DiscreteState` with a value of `1.0` at `t = 0`.

```jldoctest
julia> using StateEstimation

julia> DiscreteState(1.0, 0)
StateEstimation.DiscreteState{Float64}([1.0], 0)

julia> DiscreteState(1.0)
StateEstimation.DiscreteState{Float64}([1.0], 0)

```

We can create a `ContinuousState` in the same way -- by providing a state vector
and, optionally, a corresponding time. In the case below, we create a
2-dimensional state with a value of `[2.0, 4.0]` at `t = 1.0`.

```jldoctest
julia> using StateEstimation

julia> ContinuousState([2.0, 4.0], 1.0)
StateEstimation.ContinuousState{Float64}([2.0, 4.0], 1.0)

```

Uncertain state types require the same information as absolute states with an
additional covariance matrix to describe the uncertainty in the state vector.
The example below demonstrates the construction of an `UncertainDiscreteState`
at `t = 0` with a 2-dimensional state vector of all ones and the identity
matrix as its covariance.

```jldoctest
julia> using StateEstimation

julia> UncertainDiscreteState(ones(2), eye(2))
StateEstimation.UncertainDiscreteState{Float64}([1.0, 1.0], [1.0 0.0; 0.0 1.0], 0)

```

The next example shows construction of a scalar `UncertainContinuousState` at
`t = 0.0` with `x = 2.0` and a variance of `0.1`.

```jldoctest
julia> using StateEstimation

julia> UncertainContinuousState(2.0, 0.1)
StateEstimation.UncertainContinuousState{Float64}([2.0], [0.1], 0.0)

```

## Example 2 - Conversion

Conversion methods allow one to transform states between absolute and
uncertain forms. The method `make_uncertain` and `make_absolute` are the primary
interface for converting between these two forms.

As the name implies, `make_uncertain` converts an absolute state to an
uncertain one. This method allows the user to provide a covariance for the new
uncertain state. If not provided, the covariance defaults to all zeros. The
default behavior becomes useful later, when propagating states through
dynamic systems and observers.

The `make_absolute` method accepts an uncertain state and produces an
absolute state with the same time format. This method requires no additional
inputs and simply discards the covariance data of the input state.

The example below shows the creation of a `ContinuousState`, conversion to an
`UncertainContinuousState`, and finally a transformation back to the original
`ContinuousState`.

```jldoctest
julia> using StateEstimation

julia> absolute_state = ContinuousState(ones(2), 1.0)
StateEstimation.ContinuousState{Float64}([1.0, 1.0], 1.0)

julia> uncertain_state = make_uncertain(absolute_state, eye(2))
StateEstimation.UncertainContinuousState{Float64}([1.0, 1.0], [1.0 0.0; 0.0 1.0], 1.0)

julia> make_absolute(uncertain_state)
StateEstimation.ContinuousState{Float64}([1.0, 1.0], 1.0)

```

Another method provides a way to convert from uncertain states to absolute
states -- the `sample` method. Sampling a state provides a probabilistic
transformation from uncertain to absolute forms. This method uses the internal
covariance of the state to draw a random state from a Gaussian distribution. The
next example draws two random samples from a scalar `UncertainContinuousState`
with state `x = 2.0` and variance `0.1`.

```jldoctest
julia> using StateEstimation

julia> srand(1); # Seed the RNG to ensure expected output

julia> state = UncertainContinuousState(2.0, 0.1)
StateEstimation.UncertainContinuousState{Float64}([2.0], [0.1], 0.0)

julia> sample(state)
StateEstimation.ContinuousState{Float64}([2.09401], 0.0)

julia> sample(state)
StateEstimation.ContinuousState{Float64}([2.12092], 0.0)

```

The result of each `sample` call is a distinct, absolute state with the same
time tag as the input state.


## Example 3 - Operations

State types support a number of algebraic operators as well. In general, if any
operation that preserves a Gaussian distribution should be supported for all
state types. The example below shows scalar addition and multiplication with
a `ContinuousState`.

```jldoctest
julia> using StateEstimation

julia> state = ContinuousState(ones(2))
StateEstimation.ContinuousState{Float64}([1.0, 1.0], 0.0)

julia> state + 2
StateEstimation.ContinuousState{Float64}([3.0, 3.0], 0.0)

julia> state * 0.1
StateEstimation.ContinuousState{Float64}([0.1, 0.1], 0.0)

```

Uncertain state types behave the way one would expect when working with
Gaussian random variables. In the code below, add an `UncertainDiscreteState` by
a scalar and multiply it by both a scalar and a matrix.

```jldoctest
julia> using StateEstimation

julia> state = UncertainDiscreteState(ones(2), diagm([1.0, 2.0]))
StateEstimation.UncertainDiscreteState{Float64}([1.0, 1.0], [1.0 0.0; 0.0 2.0], 0)

julia> state + 2
StateEstimation.UncertainDiscreteState{Float64}([3.0, 3.0], [1.0 0.0; 0.0 2.0], 0)

julia> state * 0.1
StateEstimation.UncertainDiscreteState{Float64}([0.1, 0.1], [0.01 0.0; 0.0 0.02], 0)

julia> (-2*eye(2)) * state
StateEstimation.UncertainDiscreteState{Float64}([-2.0, -2.0], [4.0 0.0; 0.0 8.0], 0)

```

In-place operations such like `.=`, `.+=`, `.-=`, and `.*=` are also valid.


## Example 4 - Metrics

The `StateEstimation` module provides some state comparison metrics. These
metrics are primarily used internally, but are made available to the user as
well.

The first metric is the Euclidean `distance` method. This method simply computes
the straight-line distance between two states with any number of dimensions. The
example below shows the creation of two 3-dimensional `ContinuousState` and the
corresponding distance calculation.

```jldoctest
julia> using StateEstimation

julia> state1 = ContinuousState(zeros(3))
StateEstimation.ContinuousState{Float64}([0.0, 0.0, 0.0], 0.0)

julia> state2 = ContinuousState(ones(3))
StateEstimation.ContinuousState{Float64}([1.0, 1.0, 1.0], 0.0)

julia> distance(state1, state2)
1.7320508075688772

```

Similarly, the `mahalanobis` method computes the Mahalanobis distance between
a point and a distribution. There are two ways of calling `mahalanobis`. The
first interface requires an absolute state (the "point") and an uncertain state
(the distribution). The example below demonstrates the use of `mahalanobis` in
this way.

```jldoctest
julia> using StateEstimation

julia> state1 = DiscreteState(zeros(2))
StateEstimation.DiscreteState{Float64}([0.0, 0.0], 0)

julia> state2 = UncertainDiscreteState(ones(2), 2.5*eye(2))
StateEstimation.UncertainDiscreteState{Float64}([1.0, 1.0], [2.5 0.0; 0.0 2.5], 0)

julia> mahalanobis(state1, state2)
0.8944271909999159

```

The other interface to `mahalanobis` uses two absolute states and a
user-provided covariance. The code below demonstrates this approach.

```jldoctest
julia> using StateEstimation

julia> state1 = ContinuousState(zeros(2))
StateEstimation.ContinuousState{Float64}([0.0, 0.0], 0.0)

julia> state2 = ContinuousState(ones(2))
StateEstimation.ContinuousState{Float64}([1.0, 1.0], 0.0)

julia> mahalanobis(state1, state2, 2.5*eye(2))
0.8944271909999159

```

Note that `distance` and `mahalanobis` do not take the times of the input states
into account, so comparisons among states at different times and with different
time formats are possible. For example,

```jldoctest
julia> using StateEstimation

julia> state1 = ContinuousState(zeros(2), 1.0)
StateEstimation.ContinuousState{Float64}([0.0, 0.0], 1.0)

julia> state2 = UncertainDiscreteState(ones(2), 0.2*eye(2))
StateEstimation.UncertainDiscreteState{Float64}([1.0, 1.0], [0.2 0.0; 0.0 0.2], 0)

julia> distance(state1, state2)
1.4142135623730951

julia> mahalanobis(state1, state2)
3.1622776601683795

```
