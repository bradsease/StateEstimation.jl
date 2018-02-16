using Documenter, StateEstimation

makedocs(
    format=:html,
    sitename="StateEstimation",
    pages=[
        "Home" => "index.md",
        "Examples" => Any[
            "Working with States" => "state_examples.md",
            "Using Systems and Observers" => "system_observer_examples.md",
            "Linear Filtering" => "kalman_filter_examples.md",
            "Nonlinear Filtering" => "nonlinear_kalman_filter_examples.md",
            "Batch Estimation" => "batch_estimation_examples.md",
            #"Multi-Target Tracking" => "multi_target_examples.md"
            ],
        "All Documentation" => Any[
            "Core Components" => Any[
                "States" => "states.md",
                "Systems" => "systems.md",
                "Observers" => "observers.md"
            ],
            "Sequential Estimation" => Any[
                "Kalman Filtering" => "kalman_filter.md",
                "Multi-Target Tracking" => "multi_target.md",
            ],
            "Batch Estimation" => Any[
                "Least Squares" => "least_squares.md",
            ],
            "Simulation" => "simulation.md"
        ]
    ])
