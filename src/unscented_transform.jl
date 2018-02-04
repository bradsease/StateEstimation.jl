#
# Unscented Transform
#


"""
    UnscentedTransform()
    UnscentedTransform(alpha::Real, beta::Real, kappa::Real)

Parameters to configure an Unscented Transform.
"""
immutable UnscentedTransform
    alpha::AbstractFloat
    beta::AbstractFloat
    kappa::AbstractFloat

    function UnscentedTransform(alpha::Real, beta::Real, kappa::Real)
        new(alpha, beta, kappa)
    end
end
UnscentedTransform() = UnscentedTransform(1e-3, 2.0, 0.0)


"""
    UnscentedKalmanFilter(sys::NonlinearSystem, obs::NonlinearObserver,
                          estimate::AbstractState[, transform_params::UnscentedTransform,
                          consider_states::Vector{UInt}])

Unscented Kalman Filter type. The internal filter model currently only supports
a discrete form:

\$x_k = F(k, x_{k-1}) + w_k\$
\$y_k = H(k, x_k) + v_k\$

where \$w_k \\sim N(0, Q)\$ and \$v_k \\sim N(0, R)\$.

Construction of an UnscentedKalmanFilter requires an initial estimate.
Internally, the initial estimate is an uncertain state type. The constructor
automatically converts absolute initial estimates to uncertain states with zero
covariance.

The `consider_states` input contains a list of indices of state elements to be
considered in the filtering process and not updated.
"""
immutable UnscentedKalmanFilter{T,S<:AbstractUncertainState{T}} <:
                                                       AbstractKalmanFilter{T,S}
    sys::AbstractSystem{T}
    obs::AbstractObserver{T}
    estimate::S
    transform_params::UnscentedTransform
    consider_states::Vector{UInt16}

    function UnscentedKalmanFilter(sys::AbstractSystem{T},
                                   obs::AbstractObserver{T}, estimate::S,
                                   transform_params::UnscentedTransform,
                                   consider_states::Vector
                                   ) where {T, S<:UncertainDiscreteState{T}}
        if !allunique(consider_states)
            throw(ArgumentError("Consider state indices must be unique"))
        end
        for idx = 1:length(consider_states)
            if consider_states[idx] > length(estimate.x)
                throw(DimensionMismatch(
                    "Consider indices extend beyond length of initial state."))
            end
        end
        new{T,S}(sys, obs, estimate, transform_params, consider_states)
    end
end
UnscentedKalmanFilter(sys, obs, estimate) =
    UnscentedKalmanFilter(sys, obs, estimate, UnscentedTransform(), [])
UnscentedKalmanFilter(sys, obs, estimate, transform_params::UnscentedTransform) =
    UnscentedKalmanFilter(sys, obs, estimate, transform_params, [])
UnscentedKalmanFilter(sys, obs, estimate, consider_states::Vector) =
    UnscentedKalmanFilter(sys, obs, estimate, UnscentedTransform(), consider_states)
function UnscentedKalmanFilter(sys, obs, estimate::AbstractAbsoluteState,
                               transform_params, consider_states)
    UnscentedKalmanFilter(sys, obs, make_uncertain(estimate),
                          transform_params, consider_states)
end




"""
    compute_weights(ut::UnscentedTransform, state_length::Integer)
"""
function compute_weights(ut::UnscentedTransform, state_length::Integer)
    lambda = ut.alpha^2*(state_length + ut.kappa) - state_length
    temp = 1 / (2*(state_length + lambda))
    Wc = fill(temp, 2*state_length+1)
    Wm = fill(temp, 2*state_length+1)
    Wm[1] = lambda / (state_length + lambda)
    Wc[1] = Wm[1] + 1 - ut.alpha^2 + ut.beta
    return Wm, Wc
end


"""
    compute_sigma_points(x::AbstractUncertainState, ut::UnscentedTransform)
"""
function compute_sigma_points(x::AbstractUncertainState,
                              ut::UnscentedTransform)
    sigma_points = fill(make_absolute(x), 1)
    state_length = length(x.x)
    lambda = ut.alpha^2*(state_length + ut.kappa) - state_length
    perturbation = chol(Hermitian((state_length + lambda)*x.P))
    for idx = 1:state_length
        @inbounds push!(sigma_points, sigma_points[1] + perturbation[:,idx])
    end
    for idx = 1:state_length
        @inbounds push!(sigma_points, sigma_points[1] - perturbation[:,idx])
    end
    return sigma_points
end


"""
    update_state_from_sigma_points!(state, sigma_points, Wm, Wc)

Process predicted sigma points and weighting coefficients to perform an in-place
update of a provided uncertain state.
"""
function update_state_from_sigma_points!(state::AbstractUncertainState,
                                         sigma_points, Wm, Wc)
    state.x .= 0
    for idx = 1:length(sigma_points)
     state.x .+= Wm[idx] * sigma_points[idx].x
    end

    state.P .= 0
    for idx = 1:length(sigma_points)
     sigma_diff = sigma_points[idx].x - state.x
     state.P .+= Wc[idx] * sigma_diff * sigma_diff'
    end

    state.t = sigma_points[1].t
    return nothing
end


"""
    transform(state::AbstractUncertainState, fcn::Function, ut::UnscentedTransform)
"""
function transform(state::AbstractUncertainState, fcn::Function, ut::UnscentedTransform)
    Wm, Wc = compute_weights(ut, length(state.x))
    sigma_points = compute_sigma_points(state, ut)
    for idx = 1:length(sigma_points)
        sigma_points[idx] = fcn(sigma_points[idx])
    end
    transformed_state = make_uncertain(sigma_points[1])
    update_state_from_sigma_points!(transformed_state, sigma_points, Wm, Wc)
    return transformed_state
end


"""
    transform!(state::AbstractUncertainState, fcn!::Function, ut::UnscentedTransform)
"""
function transform!(state::AbstractUncertainState, fcn!::Function, ut::UnscentedTransform)
    Wm, Wc = compute_weights(ut, length(state.x))
    sigma_points = compute_sigma_points(state, ut)
    for idx = 1:length(sigma_points)
        fcn!(sigma_points[idx])
    end
    update_state_from_sigma_points!(state, sigma_points, Wm, Wc)
    return nothing
end


"""
    augment(state::UncertainDiscreteState, sys::AbstractSystem)
    augment(state::AbstractUncertainState, obs::AbstractObserver)
    augment(state::UncertainDiscreteState, sys::AbstractSystem, obs:AbstractObserver)

Create augmented states, systems, and observers to facilitate an unscented
transform.
"""
# TODO: Find a way to reduce duplicated code here
function augment(state::UncertainDiscreteState, sys::LinearSystem)
    n = length(state.x)
    augmented_state = expand_state(state, zeros(n), sys.Q)
    augmented_A = hvcat((2,2), sys.A, eye(n,n), zeros(n,n), eye(n,n))
    augmented_system = LinearSystem(augmented_A, zeros(size(augmented_A)))
    return augmented_state, augmented_system
end
function augment(state::UncertainDiscreteState, sys::NonlinearSystem)
    n = length(state.x)
    augmented_state = expand_state(state, zeros(n), sys.Q)
    augmented_F(t, x) = vcat(sys.F(t, x[1:n]) + x[n+1:end], x[n+1:end])
    augmented_dF(t, x) = hvcat((2,2), sys.dF_dx(t,x[1:n]), eye(n,n), zeros(n,n), eye(n,n))
    augmented_system = NonlinearSystem(augmented_F, augmented_dF, zeros(2*n,2*n))
    return augmented_state, augmented_system
end
function augment(state::AbstractUncertainState, obs::LinearObserver)
    n,m = length(state.x), size(obs.R,1)
    augmented_state = expand_state(state, zeros(m), obs.R)
    augmented_H = hcat(obs.H, eye(m,m))
    augmented_observer = LinearObserver(augmented_H, zeros(m,m))
    return augmented_state, augmented_observer
end
function augment(state::AbstractUncertainState, obs::NonlinearObserver)
    n,m = length(state.x), size(obs.R,1)
    augmented_state = expand_state(state, zeros(m), obs.R)
    augmented_H(t, x) = obs.H(t, x[1:n]) + x[n+1:end]
    augmented_dH(t, x) = hcat(obs.dH_dx(t,x[1:n]), eye(m,m))
    augmented_observer = NonlinearObserver(augmented_H, augmented_dH, zeros(m,m))
    return augmented_state, augmented_observer
end
function augment(state::UncertainDiscreteState, sys::LinearSystem, obs::LinearObserver)
    n,m = length(state.x), size(obs.R,1)
    augmented_state = expand_state(state, zeros(n+m), block_diagonal(sys.Q, obs.R))

    augmented_A = hvcat((3,2), sys.A, eye(n,n), zeros(n,m),
                               zeros(n+m,n), eye(n+m,n+m))
    augmented_system = LinearSystem(augmented_A, zeros(size(augmented_A)))

    augmented_H = hcat(obs.H, zeros(m,n), eye(m,m))
    augmented_observer = LinearObserver(augmented_H, zeros(m,m))

    return augmented_state, augmented_system, augmented_observer
end
function augment(state::UncertainDiscreteState, sys::LinearSystem, obs::NonlinearObserver)
    n,m = length(state.x), size(obs.R,1)
    augmented_state = expand_state(state, zeros(n+m), block_diagonal(sys.Q, obs.R))

    augmented_A = hvcat((3,2), sys.A, eye(n,n), zeros(n,m),
                               zeros(n+m,n), eye(n+m,n+m))
    augmented_system = LinearSystem(augmented_A, zeros(size(augmented_A)))

    augmented_H(t, x) = obs.H(t, x[1:n]) + x[2*n+1:end]
    augmented_dH(t, x) = hcat(obs.dH_dx(t,x[1:n]), zeros(m,n), eye(m,m))
    augmented_observer = NonlinearObserver(augmented_H, augmented_dH, zeros(m,m))

    return augmented_state, augmented_system, augmented_observer
end
function augment(state::UncertainDiscreteState, sys::NonlinearSystem, obs::LinearObserver)
    n,m = length(state.x), size(obs.R,1)
    augmented_state = expand_state(state, zeros(n+m), block_diagonal(sys.Q, obs.R))

    augmented_F(t, x) = vcat(sys.F(t, x[1:n]) + x[n+1:2*n], x[n+1:end])
    augmented_dF(t, x) = hvcat((3,3), sys.dF_dx(t,x[1:n]), eye(n,n), zeros(n,m),
                                      zeros(n+m,n), eye(n+m,n), zeros(n+m,m))
    augmented_system = NonlinearSystem(augmented_F, augmented_dF, zeros(2*n+m,2*n+m))

    augmented_H = hcat(obs.H, zeros(m,n), eye(m,m))
    augmented_observer = LinearObserver(augmented_H, zeros(m,m))

    return augmented_state, augmented_system, augmented_observer
end
function augment(state::UncertainDiscreteState, sys::NonlinearSystem, obs::NonlinearObserver)
    n,m = length(state.x), size(obs.R,1)
    augmented_state = expand_state(state, zeros(n+m), block_diagonal(sys.Q, obs.R))

    augmented_F(t, x) = vcat(sys.F(t, x[1:n]) + x[n+1:2*n], x[n+1:end])
    augmented_dF(t, x) = hvcat((3,3), sys.dF_dx(t,x[1:n]), eye(n,n), zeros(n,m),
                                      zeros(n+m,n), eye(n+m,n), zeros(n+m,m))
    augmented_system = NonlinearSystem(augmented_F, augmented_dF, zeros(2*n+m,2*n+m))

    augmented_H(t, x) = obs.H(t, x[1:n]) + x[2*n+1:end]
    augmented_dH(t, x) = hcat(obs.dH_dx(t,x[1:n]), zeros(m,n), eye(m,m))
    augmented_observer = NonlinearObserver(augmented_H, augmented_dH, zeros(m,m))

    return augmented_state, augmented_system, augmented_observer
end



# Unscented prediction methods
function predict!(state::UncertainDiscreteState, sys::AbstractSystem,
                 ut::UnscentedTransform, t::Integer)
    predicted_state = state
    for idx = state.t:t-1
        augmented_state, augmented_system = augment(predicted_state, sys)
        predict_wrapper!(x) = predict!(x, augmented_system, x.t+1)
        transform!(augmented_state, predict_wrapper!, ut)
        predicted_state = reduce_state(augmented_state, length(state.x))
    end
    state.x .= predicted_state.x
    state.P .= predicted_state.P
    state.t = predicted_state.t
    return nothing
end
function predict(state::UncertainDiscreteState, sys::AbstractSystem,
                 ut::UnscentedTransform, t::Integer)
    predicted_state = deepcopy(state)
    predict!(predicted_state, sys, ut, t)
    return predicted_state
end
function predict(state::AbstractUncertainState, obs::AbstractObserver,
                 ut::UnscentedTransform)
    augmented_state, augmented_observer = augment(state, obs)
    predict_wrapper(x) = predict(x, augmented_observer)
    augmented_prediction = transform(augmented_state, predict_wrapper, ut)
    return reduce_state(augmented_prediction, size(obs.R,1))
end


"""
    kalman_predict(ukf::UnscentedKalmanFilter, t)
"""
function kalman_predict(ukf::UnscentedKalmanFilter, t)

    # Predict to t-1, if necesary
    if (t > ukf.estimate.t+1)
        predict!(ukf.estimate, ukf.sys, ukf.transform_params, t-1)
    end

    # Set up unscented transform
    augmented_state, augmented_system, augmented_observer =
        augment(ukf.estimate, ukf.sys, ukf.obs)
    Wm, Wc = compute_weights(ukf.transform_params, length(augmented_state.x))
    sigma_points = compute_sigma_points(augmented_state, ukf.transform_params)

    # Predict sigma points through system
    fcn!(x) = predict!(x, augmented_system, x.t+1)
    for idx = 1:length(sigma_points)
        fcn!(sigma_points[idx])
    end

    # Predict sigma points through observer
    fcn(x) = predict(x, augmented_observer)
    measurement_sigma_points = [fcn(x) for x in sigma_points]

    # Compute xk
    augmented_xk = make_uncertain(sigma_points[1])
    update_state_from_sigma_points!(augmented_xk, sigma_points, Wm, Wc)
    xk = reduce_state(augmented_xk, length(ukf.estimate.x))

    # Compute yk
    yk = make_uncertain(measurement_sigma_points[1])
    update_state_from_sigma_points!(yk, measurement_sigma_points, Wm, Wc)

    # Compute Pxy
    n = length(xk.x)
    Pxy = zeros(length(xk.x), length(yk.x))
    for idx = 1:length(sigma_points)
        Pxy .+= Wc[idx] * (sigma_points[idx].x[1:n] - xk.x) *
                (measurement_sigma_points[idx].x - yk.x)'
    end

    return xk, yk, Pxy
end
