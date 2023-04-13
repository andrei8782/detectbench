import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import seaborn as sns


def plot_snrs_per_unit_group(snrs_per_unit_group):
    # Prepare data for the plot
    channel_groups = list(snrs_per_unit_group.keys())
    mean_snrs = [snr_data[1] for snr_data in snrs_per_unit_group.values()]
    unit_indices = [snr_data[0] for snr_data in snrs_per_unit_group.values()]

    plt.figure(figsize=(10, 5))
    x_positions = np.arange(len(channel_groups))
    bars = plt.bar(x_positions, mean_snrs, tick_label=channel_groups)

    plt.xlabel("Extremum Channel Group")
    plt.ylabel("Mean SNR")
    plt.title("Mean SNR per Ground Truth Spike Units Group")

    # Create custom x-axis labels
    x_labels = [
        f"CH {ch}: {','.join(map(str, units))}"
        for ch, units in zip(channel_groups, unit_indices)
    ]
    plt.xticks(x_positions, x_labels, rotation=45, ha="right")

    plt.show()


def plot_metric_per_extremum_channel(
    metric_values: Dict[int, float], metric_name: str
):
    # Prepare data for the plot
    channel_groups = list(metric_values.keys())
    metrics = list(metric_values.values())

    plt.figure(figsize=(10, 5))
    x_positions = np.arange(len(channel_groups))
    bars = plt.bar(x_positions, metrics, tick_label=channel_groups)

    x_labels = [f"CH {ch}" for ch in channel_groups]
    plt.xticks(x_positions, x_labels)
    plt.xlabel("GT Units mapped with Extremum Channel")
    plt.ylabel(f"{metric_name} Value")
    plt.title(f"{metric_name} Value per Extremum Channel Group")

    plt.show()


# The next methods require manual data entry
# of the metrics computed during the experiments.
# They serve as a reference and should be automated in the future.


def plot_metrics(thresholds, accuracies, precisions, recalls):
    plt.plot(thresholds, accuracies, label="accuracy", marker="o", color="g")
    plt.plot(thresholds, precisions, label="precision", marker="s", color="r")
    plt.plot(thresholds, recalls, label="recall", marker="x", color="b")
    plt.xlabel("Threshold")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def plot_unoptimised_optimised_bar():
    sns.set(style="whitegrid", context="paper", font_scale=1.3)

    algorithms = [
        "Herding Spikes 2.1",
        "Herding Spikes 2",
        "Tridesclous CH",
        "Tridesclous LE",
    ]
    before_optimisation = [0.62, 0.66, 0.34, 0.55]
    after_optimisation = [0.75, 0.78, 0.50, 0.72]

    n_algorithms = len(algorithms)
    ind = np.arange(n_algorithms)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_before = ax.bar(
        ind,
        before_optimisation,
        width,
        color="#fc8d62",
        label="Before Optimisation",
    )
    bars_after = ax.bar(
        ind,
        [
            after - before
            for after, before in zip(after_optimisation, before_optimisation)
        ],
        width,
        bottom=before_optimisation,
        color="#8da0cb",
        label="After Optimisation",
    )

    unique_values = sorted(set(before_optimisation + after_optimisation + [0]))
    ax.set_yticks(unique_values)
    ax.set_yticklabels([f"{tick}" for tick in unique_values])

    ax.set_title("Accuracy Before and After Optimisation")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(ind)
    ax.set_xticklabels(algorithms)
    ax.legend(bbox_to_anchor=(0.72, 1.017), loc="upper left")
    # plt.savefig("optimised_vs_unoptimised_acc2.pdf", dpi=900)
    plt.show()


def plot_unoptimised_optimised_bar2():
    algorithms = ["HS 2.1", "HS 2", "Tridesclous CH", "Tridesclous LE"]
    before_optimisation = [0.62, 0.66, 0.34, 0.55]
    after_optimisation = [0.75, 0.78, 0.50, 0.72]
    running_times = [28, 144, 35, 31]

    n_algorithms = len(algorithms)
    ind = np.arange(n_algorithms)
    width = 0.5

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 6), sharey=False)

    # Plot the accuracy comparison
    bars_before = ax.bar(
        ind,
        before_optimisation,
        width,
        color="#fc8d62",
        label="Before Optimisation",
    )
    bars_after = ax.bar(
        ind,
        [
            after - before
            for after, before in zip(after_optimisation, before_optimisation)
        ],
        width,
        bottom=before_optimisation,
        color="#8da0cb",
        label="After Optimisation",
    )

    unique_values = sorted(set(before_optimisation + after_optimisation + [0]))
    ax.set_yticks(unique_values)
    ax.set_yticklabels([f"{tick}" for tick in unique_values])

    ax.set_title("Accuracy Before and After Optimisation")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(ind)
    ax.set_xticklabels(algorithms, rotation=-45)
    # ax.legend(bbox_to_anchor=(0.72, 1.017), loc='upper left')
    # Make the legend smaller
    ax.legend(bbox_to_anchor=(0.6, 1.017), loc="upper left", prop={"size": 10})
    # Plot the running time comparison
    bars_time = ax2.bar(
        ind, running_times, width, color="#66c2a5", label="Running Time"
    )

    ax2.set_title("Detection Running Time Comparison")
    ax2.set_xlabel("Algorithm")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_xticks(ind)
    ax2.set_xticklabels(algorithms, rotation=-45)

    plt.tight_layout()
    # plt.savefig("optimised_vs_unoptimised_acc2_and_time_tilted_labels.pdf", dpi=900)
    plt.show()


def plot_unoptimised_optimised_bar3():
    algorithms = ["Herding Spikes 2.1", "Herding Spikes 2", "Tridesclous LE"]
    before_optimisation = [0.0157, 0.0127, 0.0392]
    after_optimisation = [0.0082, 0.0102, 0.021]
    acc_before_optimisation = [0.75, 0.78, 0.72]
    acc_after_optimisation = [9.7, 9.9, 8.1]

    n_algorithms = len(algorithms)
    ind = np.arange(n_algorithms)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use Orange and Blue colors from the Set2 color scheme
    bars_before = ax.bar(
        ind,
        before_optimisation,
        width,
        color="#66c2a5",
        label="After the second optimisation (by precision)",
    )
    bars_after = ax.bar(
        ind,
        [
            after - before
            for after, before in zip(after_optimisation, before_optimisation)
        ],
        width,
        bottom=before_optimisation,
        color="#8da0cb",
        label="After the first optimisation (by accuracy)",
    )

    # Add time labels
    for i, (bar_before, bar_after) in enumerate(zip(bars_before, bars_after)):
        ax.text(
            bar_before.get_x() + bar_before.get_width() / 2,
            bar_before.get_height() - 0.001,
            f"accuracy={acc_before_optimisation[i]}",
            ha="center",
            va="top",
            fontsize=10,
        )
        ax.text(
            bar_after.get_x() + bar_after.get_width() / 2,
            bar_after.get_y() + bar_after.get_height() - 0.001,
            f"accuracy loss \n after opt \n {acc_after_optimisation[i]}%",
            ha="center",
            va="top",
            fontsize=10,
        )

    # Show Y-axis ticks with 0, 1, and unique values from both before_optimisation and after_optimisation
    unique_values = sorted(set(before_optimisation + after_optimisation + [0]))
    ax.set_yticks(unique_values)
    ax.set_yticklabels([f"{tick:.2%}" for tick in unique_values])

    ax.set_title("Duplicate Spikes % Before and After The Second Optimisation")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Duplicate Spikes %")
    ax.set_xticks(ind)
    ax.set_xticklabels(algorithms)
    ax.legend()
    # plt.savefig("duplicate_percentage.pdf", dpi=600)
    plt.show()
