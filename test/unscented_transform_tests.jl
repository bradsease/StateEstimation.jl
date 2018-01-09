# Unscented Transform


# Test constructors
@test UnscentedTransform() == UnscentedTransform(1e-3, 2, 0)

# Test methods
state = UncertainContinuousState(ones(3), diagm([1.0, 2.0, 3.0]))
transform = UnscentedTransform()
compute_weights(transform, 3)
compute_sigma_points(transform, state)


#
state = UncertainContinuousState(1.0, 0.1)
sys = LinearSystem(2.0)
predict(state, sys, transform, 1.0)
#println(predict(state, sys, transform, 1.0))
#println(predict(state, sys, 1.0))
