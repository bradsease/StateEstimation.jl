using StateEstimation
using Base.Test

# Sanity check
@test 1 == 1


# Create linear observer
linear_sys = LinearSystem(0.5*eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.001*eye(2))
initial_est = UncertainDiscreteState([1.0, 2.0], 0.1*eye(2))
kf1 = KalmanFilter(linear_sys, linear_obs, initial_est)


# Create linear observer
linear_sys = LinearSystem(-eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.001*eye(2))
initial_est = UncertainContinuousState([1.0, 2.0], 0.1*eye(2))
kf2 = KalmanFilter(linear_sys, linear_obs, initial_est)


mtf = NearestNeighborMTF(kf1)
#add!(mtf, kf1)
add!(mtf, kf1)
#add!(mtf, [kf1, kf2])
process!(mtf, simulate(mtf.filter_bank[1], mtf.filter_bank[1].estimate.t+1))
