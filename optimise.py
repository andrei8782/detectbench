from benchmark import run_benchmark
import numpy as np
from skopt import Optimizer
import matplotlib.pyplot as plt
from skopt.plots import plot_objective
from skopt.space import Space


class OptimizerResultWrapper:
    def __init__(self, optimizer, space):
        self.optimizer = optimizer
        self.space = space
        self.x = optimizer.Xi[np.argmin(optimizer.yi)]
        self.fun = np.min(optimizer.yi)
        self.x_iters = optimizer.Xi
        self.func_vals = optimizer.yi
        self.models = optimizer.models


class DetectionOptimizer:
    def __init__(
        self,
        recording,
        extr_ch_to_units_trains,
        detection_method,
        detection_kwargs,
        detection_config_kwargs,
        detection_job_kwargs,
        benchmark_config_kwargs,
    ):
        self.detection_kwargs = detection_kwargs
        self.detection_config_kwargs = detection_config_kwargs
        self.detection_job_kwargs = detection_job_kwargs
        self.benchmark_config_kwargs = benchmark_config_kwargs
        self.recording = recording
        self.extr_ch_to_units_trains = extr_ch_to_units_trains
        self.detection_method = detection_method

    def plot_convergence(self):
        if hasattr(self, "optimizer"):
            yi = np.array(self.optimizer.yi)
            min_yi = np.minimum.accumulate(yi)

            # Custom convergence plot
            plt.plot(range(1, len(min_yi) + 1), min_yi, marker="o")
            plt.xlabel("Number of calls")
            plt.ylabel("Best score obtained")
            plt.title("Convergence plot")
            plt.grid()
            plt.show()
        else:
            print("No optimization results available.")

    def plot_objective(self, space):
        if hasattr(self, "optimizer"):
            space_object = Space(space)
            result_wrapper = OptimizerResultWrapper(
                self.optimizer, space_object
            )
            plot_objective(result_wrapper)
            plt.show()
        else:
            print("No optimization has been performed yet.")

    def objective(self, params, space):
        param_dict = {
            dimension.name: value for dimension, value in zip(space, params)
        }

        for key, value in param_dict.items():
            self.detection_kwargs[key] = value

        benchmarking_log = run_benchmark(
            recording=self.recording,
            extr_ch_to_units_trains=self.extr_ch_to_units_trains,
            detection_method=self.detection_method,
            detection_kwargs=self.detection_kwargs,
            detection_config_kwargs=self.detection_config_kwargs,
            detection_job_kwargs=self.detection_job_kwargs,
            **self.benchmark_config_kwargs,
        )

        metric_to_optimize = (
            "accuracy"
        )  # Can be "accuracy", "precision", or "recall"
        return -benchmarking_log[metric_to_optimize]

    def optimize(
        self,
        space,
        fixed_hyperparams,
        n_iterations,
        early_stopping_tol=0.001,
        early_stopping_patience=10,
    ):
        # Set a fixed random_state for reproducibility
        self.optimizer = Optimizer(
            space, base_estimator="GP", n_initial_points=10, random_state=42
        )

        best_score = float("inf")
        patience_counter = 0

        for i in range(n_iterations):
            next_params = self.optimizer.ask()
            for key, value in fixed_hyperparams.items():
                self.detection_kwargs[key] = value
            f_val = self.objective(next_params, space)
            self.optimizer.tell(next_params, f_val)
            print(
                f"Iteration {i+1}/{n_iterations}: {next_params} -> {abs(f_val)}"
            )

            # Early stopping functionality
            if abs(f_val - best_score) < early_stopping_tol:
                patience_counter += 1
            else:
                patience_counter = 0
            best_score = min(best_score, f_val)

            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

        best_params = self.optimizer.Xi[np.argmin(self.optimizer.yi)]
        return best_params, best_score
