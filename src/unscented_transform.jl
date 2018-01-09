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
    compute_weights(transform::UnscentedTransform, state_length::Integer)
"""
function compute_weights(transform::UnscentedTransform, state_length::Integer)
    lambda = transform.alpha^2*(state_length + transform.kappa) - state_length
    temp = 1 / (2*state_length + lambda)
    Wc = fill(temp, 2*state_length+1)
    Wm = fill(temp, 2*state_length+1)
    Wm[1] = lambda / (state_length + lambda)
    Wc[1] = Wm[1] + 1 - transform.alpha^2 - transform.beta
    return Wm, Wc
end


"""
    compute_sigma_points(transform::UnscentedTransform, x0::AbstractUncertainState)
"""
function compute_sigma_points(transform::UnscentedTransform,
                              x0::AbstractUncertainState)
    sigma_points = fill(make_absolute(x0), 1)
    state_length = length(x0.x)
    lambda = transform.alpha^2*(state_length + transform.kappa) - state_length
    perturbation = chol((state_length + lambda)*x0.P)
    for idx = 1:state_length
        @inbounds push!(sigma_points, sigma_points[1] + perturbation[:,idx])
    end
    for idx = 1:state_length
        @inbounds push!(sigma_points, sigma_points[1] - perturbation[:,idx])
    end
    return sigma_points
end


function predict(state::AbstractUncertainState, sys::AbstractSystem,
                 transform::UnscentedTransform, t::Real)
    predicted_state = deepcopy(state)
    predict!(state, sys, transform, t)
    return predicted_state
end
function predict!(state::AbstractUncertainState, sys::AbstractSystem,
                 transform::UnscentedTransform, t::Real)
    Wm, Wc = compute_weights(transform, length(state.x))
    sigma_points = compute_sigma_points(transform, state)
    for idx = 1:length(sigma_points)
        predict!(sigma_points[idx], sys, t)
    end

    predicted_state = deepcopy(state)
    predicted_state.x .= 0
    for idx = 1:length(sigma_points)
        predicted_state.x .+= Wm[idx] * sigma_points[idx].x
    end

    predicted_state.P .= 0
    for idx = 1:length(sigma_points)
        sigma_diff = sigma_points[idx].x - predicted_state.x
        predicted_state.P .+= Wc[idx] * sigma_diff * sigma_diff'
    end

    predicted_state.t = t
    return predicted_state
end
