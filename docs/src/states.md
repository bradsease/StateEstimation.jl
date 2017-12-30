# State Types and Methods

## Absolute States

```@docs
ContinuousState
```

```@docs
DiscreteState
```


## Uncertain States

```@docs
UncertainContinuousState
```

```@docs
UncertainDiscreteState
```


## Methods

```@docs
sample
```

```@docs
distance{T}(::AbstractState{T}, ::AbstractState{T})
```

```@docs
mahalanobis
```

```@docs
make_uncertain
```

```@docs
make_absolute
```

## Operators

State types also support a limited number of algebraic operators. Since States
usually represent a Gaussian distribution, operations that preserve the
that distribution are typically supported.

For example,
```jldoctest
julia> using StateEstimation

julia> test_state = UncertainContinuousState([1.0, 2.0], eye(2), 0.0)
StateEstimation.UncertainContinuousState{Float64}([1.0, 2.0], [1.0 0.0; 0.0 1.0], 0.0)

julia> test_state + 2
StateEstimation.UncertainContinuousState{Float64}([3.0, 4.0], [1.0 0.0; 0.0 1.0], 0.0)

julia> test_state * 2
StateEstimation.UncertainContinuousState{Float64}([2.0, 4.0], [4.0 0.0; 0.0 4.0], 0.0)
```
