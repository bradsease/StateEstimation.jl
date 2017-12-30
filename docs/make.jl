using Documenter, StateEstimation

makedocs(
    format=:html,
    sitename="StateEstimation",
    pages=[
        "Home" => "index.md",
        "Examples" => Any[
            "Working with States" => "state_examples.md",
            "Using Systems and Observers" => "system_observer_examples.md",
            "Kalman Filtering" => "kalman_filter_examples.md",
            "Batch Estimation" => "batch_estimation_examples.md",
            "Multi-Target Tracking" => "multi_target_examples.md"
            ],
        "All Documentation" => Any[
            "Core Components" => Any[
                "States" => "states.md",
                "Systems" => "systems.md",
                "Observers" => "observers.md"
            ],
            "Sequential Estimators" => Any[
                "Kalman Filter" => "kalman_filter.md",
                "Extended Kalman Filter" => "extended_kalman.md",
                "Unscented Kalman Filter" => "unscented_kalman.md"
            ],
            "Batch Estimators" => Any[
                "Least Squares" => "least_squares.md",
            ]
        ]
    ])
