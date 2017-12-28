# Multi Target Filter tests


## Continuous filter tests

# Create filter 1
linear_sys = LinearSystem(0.5*eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.001*eye(2))
initial_est = UncertainContinuousState([0.0, 0.0], 0.1*eye(2))
kf1 = KalmanFilter(linear_sys, linear_obs, initial_est)

# Create filter 2
linear_sys = LinearSystem(-eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.001*eye(2))
initial_est = UncertainContinuousState([1.0, 2.0], 0.1*eye(2))
kf2 = KalmanFilter(linear_sys, linear_obs, initial_est)

# Create filter 3
linear_sys = LinearSystem(-eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.001*eye(2))
initial_est = UncertainContinuousState([10.0, 20.0], 0.1*eye(2))
kf3 = KalmanFilter(linear_sys, linear_obs, initial_est)

# Build MTF
mtf = NearestNeighborMTF([kf1])
add!(mtf, [kf2, kf3])
@test_throws ArgumentError NearestNeighborMTF([])
@test_throws ArgumentError add!(mtf, kf2)

# Simulate data and process
srand(1);
for i = 1:10
    for (j, measurement) in enumerate(inaccurate_simulate(mtf, i*0.1))
        process!(mtf, measurement)
        @test mtf.filter_bank[j].estimate.t == i*0.1
    end
end


## Discrete filter tests

# Create filter 1
linear_sys = LinearSystem(0.5*eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.001*eye(2))
initial_est = UncertainDiscreteState([0.0, 0.0], 0.1*eye(2))
kf1 = KalmanFilter(linear_sys, linear_obs, initial_est)

# Create filter 2
linear_sys = LinearSystem(-eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.001*eye(2))
initial_est = UncertainDiscreteState([1.0, 2.0], 0.1*eye(2))
kf2 = KalmanFilter(linear_sys, linear_obs, initial_est)

# Create filter 3
linear_sys = LinearSystem(-0.9*eye(2), 0.001*eye(2))
linear_obs = LinearObserver(eye(2), 0.001*eye(2))
initial_est = UncertainDiscreteState([10.0, 20.0], 0.1*eye(2))
kf3 = KalmanFilter(linear_sys, linear_obs, initial_est)

# Build MTF
mtf = NearestNeighborMTF([kf1, kf2, kf3])
@test_throws ArgumentError NearestNeighborMTF([])
@test_throws ArgumentError add!(mtf, kf2)

# Simulate data and process
srand(1);
for i = 1:10
    for (j, measurement) in enumerate(inaccurate_simulate(mtf, i))
        process!(mtf, measurement)
        @test mtf.filter_bank[j].estimate.t == i
    end
end
