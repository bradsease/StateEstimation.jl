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
    compute_sigma_points(x::Vector, P::Matrix, ut::UnscentedTransform)
    compute_sigma_points(x::AbstractUncertainState, ut::UnscentedTransform)
"""
function compute_sigma_points(x::AbstractUncertainState,
                              ut::UnscentedTransform)
    sigma_points = fill(make_absolute(x), 1)
    state_length = length(x.x)
    lambda = ut.alpha^2*(state_length + ut.kappa) - state_length
    perturbation = chol((state_length + lambda)*x.P)
    for idx = 1:state_length
        @inbounds push!(sigma_points, sigma_points[1] + perturbation[:,idx])
    end
    for idx = 1:state_length
        @inbounds push!(sigma_points, sigma_points[1] - perturbation[:,idx])
    end
    return sigma_points
end
function compute_sigma_points(x::Vector, P::Matrix, ut::UnscentedTransform)
    sigma_points = fill(deepcopy(x), 1)
    state_length = length(x)
    lambda = ut.alpha^2*(state_length + ut.kappa) - state_length
    perturbation = chol((state_length + lambda)*P)
    for idx = 1:state_length
        @inbounds push!(sigma_points, sigma_points[1] + perturbation[:,idx])
    end
    for idx = 1:state_length
        @inbounds push!(sigma_points, sigma_points[1] - perturbation[:,idx])
    end
    return sigma_points
end


"""
    transform!(x::Vector, P::Matrix, fcn!::Function, ut::UnscentedTransform)
    transform!(state::AbstractUncertainState, fcn!::Function, ut::UnscentedTransform)
"""
function transform(state::AbstractUncertainState, fcn!::Function, ut::UnscentedTransform)
    transformed_state = deepcopy(state)
    transform!(transformed_state, fcn!, ut)
    return transformed_state
end
function transform(x::Vector, P::Matrix, fcn!::Function, ut::UnscentedTransform)
    transformed_x = deepcopy(x)
    transformed_P = deepcopy(P)
    transform!(transformed_x, transformed_P, fcn!, ut)
    return transformed_x, transformed_P
end


"""
    transform!(x::Vector, P::Matrix, fcn!::Function, ut::UnscentedTransform)
    transform!(state::AbstractUncertainState, fcn!::Function, ut::UnscentedTransform)
"""
function transform!(state::AbstractUncertainState, fcn!::Function, ut::UnscentedTransform)
    Wm, Wc = compute_weights(ut, length(state.x))
    sigma_points = compute_sigma_points(state, ut)
    for idx = 1:length(sigma_points)
        fcn!(sigma_points[idx])
    end

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
function transform!(x::Vector, P::Matrix, fcn!::Function, ut::UnscentedTransform)
    Wm, Wc = compute_weights(ut, length(x))
    sigma_points = compute_sigma_points(x, P, ut)
    for idx = 1:length(sigma_points)
        fcn!(sigma_points[idx])
    end

    x .= 0
    for idx = 1:length(sigma_points)
        x .+= Wm[idx] * sigma_points[idx]
    end

    P .= 0
    for idx = 1:length(sigma_points)
        sigma_diff = sigma_points[idx] - x
        P .+= Wc[idx] * sigma_diff * sigma_diff'
    end
    return nothing
end

"""
"""
function grow_state(state::AbstractUncertainState, x::Vector, P::Matrix)
    n = length(x)
    if size(P) != (n,n)
        throw(DimensionMismatch("Dimensions of x and P are inconsistent."))
    end
    augmented_state = deepcopy(state)
    augmented_state.x = vcat(state.x, x)
    augmented_state.P = block_diagonal(state.P, P)
    return augmented_state
end

"""
Simple blkdiag alternative for two non-sparse matrices.
"""
function block_diagonal(A1, A2)
    m1, n1 = size(A1)
    m2, n2 = size(A2)
    A_out = zeros(m1+m2, n1+n2)
    A_out[1:m1, 1:n1] .= A1
    A_out[m1+1:end, n1+1:end] .= A2
    return A_out
end


"""
    augment(state::AbstractUncertainState, sys::AbstractSystem)
    augment(state::AbstractUncertainState, obs::AbstractObserver)
    augment(state::AbstractUncertainState, sys::AbstractSystem, obs:AbstractObserver)

Create augmented states, systems, and observers to facilitate an unscented
transform.
"""
function augment(state::UncertainContinuousState, sys::LinearSystem)
    n = length(state.x)
    augmented_state = grow_state(state, zeros(n), sys.Q)
    augmented_A = hvcat((2,1), sys.A, eye(n,n), zeros(n, 2*n))
    augmented_system = LinearSystem(augmented_A, zeros(size(augmented_A)))
    return augmented_state, augmented_system
end
function augment(state::UncertainDiscreteState, sys::LinearSystem)
    n = length(state.x)
    augmented_state = grow_state(state, zeros(n), sys.Q)
    augmented_A = hvcat((2,2), sys.A, eye(n,n), zeros(n,n), eye(n,n))
    augmented_system = LinearSystem(augmented_A, zeros(size(augmented_A)))
    return augmented_state, augmented_system
end

function augment(state::UncertainContinuousState, sys::NonlinearSystem)
    n = length(state.x)
    augmented_state = grow_state(state, zeros(n), sys.Q)
    augmented_F(t, x) = vcat(sys.F(t, x[1:n]) + x[n+1:end], zeros(n))
    augmented_dF(t, x) = hvcat((2,1), sys.dF_dx(t,x[1:n]), eye(n,n), zeros(n,2*n))
    augmented_system = NonlinearSystem(augmented_F, augmented_dF, zeros(2*n,2*n))
    return augmented_state, augmented_system
end
function augment(state::UncertainDiscreteState, sys::NonlinearSystem)
    n = length(state.x)
    augmented_state = grow_state(state, zeros(n), sys.Q)
    augmented_F(t, x) = vcat(sys.F(t, x[1:n]) + x[n+1:end], x[n+1:end])
    augmented_dF(t, x) = hvcat((2,2), sys.dF_dx(t,x[1:n]), eye(n,n), zeros(n,n), eye(n,n))
    augmented_system = NonlinearSystem(augmented_F, augmented_dF, zeros(2*n,2*n))
    return augmented_state, augmented_system
end

function augment(state::AbstractUncertainState, obs::LinearObserver)
    n,m = length(state.x), size(obs.R,1)
    augmented_state = grow_state(state, zeros(m), obs.R)
    augmented_H = hcat(obs.H, eye(m,m))
    augmented_observer = LinearObserver(augmented_H, zeros(m,m))
    return augmented_state, augmented_observer
end
function augment(state::AbstractUncertainState, obs::NonlinearObserver)
    n,m = length(state.x), size(obs.R,1)
    augmented_state = grow_state(state, zeros(m), obs.R)
    augmented_H(t, x) = obs.H(t, x[1:n]) + x[n+1:end]
    augmented_dH(t, x) = hcat(obs.dH_dx(t,x[1:n]), eye(m,m))
    augmented_observer = NonlinearObserver(augmented_H, augmented_dH, zeros(m,m))
    return augmented_state, augmented_observer
end

#
# TODO: Find a way to reduce duplicated code here (and above)
#
function augment(state::UncertainContinuousState, sys::LinearSystem, obs::LinearObserver)
    n,m = length(state.x), size(obs.R,1)
    augmented_state = grow_state(state, zeros(n+m), block_diagonal(sys.Q, obs.R))

    augmented_A = hvcat((3,1), sys.A, eye(n,n), zeros(n,m), zeros(n+m,2*n+m))
    augmented_system = LinearSystem(augmented_A, zeros(size(augmented_A)))

    augmented_H = hcat(obs.H, zeros(m,n), eye(m,m))
    augmented_observer = LinearObserver(augmented_H, zeros(m,m))

    return augmented_state, augmented_system, augmented_observer
end
function augment(state::UncertainDiscreteState, sys::LinearSystem, obs::LinearObserver)
    n,m = length(state.x), size(obs.R,1)
    augmented_state = grow_state(state, zeros(n+m), block_diagonal(sys.Q, obs.R))

    augmented_A = hvcat((3,3), sys.A, eye(n,n), zeros(n,m),
                               zeros(n+m,n), eye(n+m,n), zeros(n+m,m))
    augmented_system = LinearSystem(augmented_A, zeros(size(augmented_A)))

    augmented_H = hcat(obs.H, zeros(m,n), eye(m,m))
    augmented_observer = LinearObserver(augmented_H, zeros(m,m))

    return augmented_state, augmented_system, augmented_observer
end
function augment(state::UncertainContinuousState, sys::LinearSystem, obs::NonlinearObserver)
    n,m = length(state.x), size(obs.R,1)
    augmented_state = grow_state(state, zeros(n+m), block_diagonal(sys.Q, obs.R))

    augmented_A = hvcat((3,1), sys.A, eye(n,n), zeros(n,m), zeros(n+m,2*n+m))
    augmented_system = LinearSystem(augmented_A, zeros(size(augmented_A)))

    augmented_H(t, x) = obs.H(t, x[1:n]) + x[2*n+1:end]
    augmented_dH(t, x) = hcat(obs.dH_dx(t,x[1:n]), zeros(m,n), eye(m,m))
    augmented_observer = NonlinearObserver(augmented_H, augmented_dH, zeros(m,m))

    return augmented_state, augmented_system, augmented_observer
end
function augment(state::UncertainDiscreteState, sys::LinearSystem, obs::NonlinearObserver)
    n,m = length(state.x), size(obs.R,1)
    augmented_state = grow_state(state, zeros(n+m), block_diagonal(sys.Q, obs.R))

    augmented_A = hvcat((3,3), sys.A, eye(n,n), zeros(n,m),
                               zeros(n+m,n), eye(n+m,n), zeros(n+m,m))
    augmented_system = LinearSystem(augmented_A, zeros(size(augmented_A)))

    augmented_H(t, x) = obs.H(t, x[1:n]) + x[2*n+1:end]
    augmented_dH(t, x) = hcat(obs.dH_dx(t,x[1:n]), zeros(m,n), eye(m,m))
    augmented_observer = NonlinearObserver(augmented_H, augmented_dH, zeros(m,m))

    return augmented_state, augmented_system, augmented_observer
end



function predict(state::AbstractUncertainState, sys::AbstractSystem,
                 ut::UnscentedTransform, t::Real)
    predicted_state = deepcopy(state)
    predict!(predicted_state, sys, ut, t)
    return predicted_state
end
function predict!(state::AbstractUncertainState, sys::AbstractSystem,
                  ut::UnscentedTransform, t::Real)
    fcn!(x) = predict!(x, sys, t)
    transform!(state, fcn!, ut)
end
