"""Auxiliary functions for notebook on hyperbolic discounting."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import respy as rp

def plot_robinson_choices_base(df, df_beta, df_lowbeta):

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot()

    x = np.arange(10)

    bar_width = 0.25

    colors = [["#1f77b4", "#ff7f0e"],
              ["#428dc1", "#ff9833"],
              ["#70a8d0", "#ffb369"]]
    labels = ["β=1", "β=0.8", "β=0.5"]
    positions = [x, x+bar_width*1.2, x+bar_width*2.4]

    for i, series in enumerate([df, df_beta, df_lowbeta]):
        hammock = series.groupby("Period").Choice.value_counts().unstack().loc[:, "hammock"]
        fishing = series.groupby("Period").Choice.value_counts().unstack().loc[:, "fishing"]
        ax.bar(
            positions[i], fishing, width=bar_width,
            color=colors[i][0], label=labels[i]
            )
        ax.bar(
            positions[i], hammock, width=bar_width,
            bottom=fishing, color=colors[i][1], label=labels[i]
            )

    ax.set_xticks(x + 2*bar_width / 2)
    ax.set_xticklabels(np.arange(10))
    ax.set_xlabel("Periods")

    handles, _ = ax.get_legend_handles_labels()
    handles_positions = [[0, 2, 4], [1, 3, 5]]
    bbox_to_anchor=[(1.12, 0.5), (1.12, 0.8)]

    for i, title in enumerate(["fishing", "hammock"]):
        legend = plt.legend(handles=list(handles[j] for j in handles_positions[i]),
        ncol=1, bbox_to_anchor=bbox_to_anchor[i], title=title, frameon=False)
        plt.gca().add_artist(legend)

    fig.suptitle("Robinson's choices", y = 0.95)

    plt.show()


def plot_return_experience(mean_max_exp_fishing_by_beta, grid_points):

    fig, ax = plt.subplots(figsize=(8,6))
    labels = ["β=1", "β=0.8", "β=0.5"]

    for mean_max_exp_fishing, label in zip(mean_max_exp_fishing_by_beta, labels):
        plt.plot(grid_points, mean_max_exp_fishing, label=label)

    plt.ylim([0, 10])
    plt.xlabel("Return to experience")
    plt.ylabel("Average final level of experience")

    plt.legend()

    plt.show()


def plot_myopic_vs_present_bias(df_myopic, df_lowbeta):

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    df_myopic.groupby("Period").Choice.value_counts().unstack().plot.bar(
        ax=axs[0], stacked=True, rot=0, legend=False, title="Completely myiopic")
    df_lowbeta.groupby("Period").Choice.value_counts().unstack().plot.bar(
        ax=axs[1], stacked=True, rot=0, title="With present bias (β=0.5)")

    handles, _ = axs[0].get_legend_handles_labels()
    axs[1].get_legend().remove()
    fig.legend(
        handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3
        )
    fig.suptitle("Robinson's choices", fontsize=14, y=1.05)

    plt.tight_layout(rect=[0,0.05,1,1])

    plt.show()


def plot_robinson_choices_extended(df_ext, df_beta_ext):

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    df_ext.groupby("Period").Choice.value_counts().unstack().plot.bar(
        ax=axs[0], stacked=True, rot=0, legend=False,
        title="Without present bias", color=["C0", "C2", "C1"])
    df_beta_ext.groupby("Period").Choice.value_counts().unstack().plot.bar(
        ax=axs[1], stacked=True, rot=0,
        title="With present bias (β=0.8)", color=["C0", "C2", "C1"])

    handles, _ = axs[0].get_legend_handles_labels()
    axs[1].get_legend().remove()
    fig.legend(
        handles=handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 0),
        ncol=3)
    fig.suptitle("Robinson's choices", fontsize=14, y=1.05)

    plt.tight_layout(rect=[0,0.05,1,1])


def plot_profile_likelihood(results, params, estimates):
    for index, fvals in results.items():
        fig, ax = plt.subplots()

        upper, lower = params.loc[index][["upper", "lower"]]
        grid = np.linspace(lower, upper, 20)

        ax.axvline(
            params.loc[index, "value"],
            color="#A9A9A9",
            linestyle="--",
            label="Baseline"
            )
        ax.axvline(
            estimates.loc[index, "value"],
            color="red",
            linestyle="--",
            label="True value")
        ax.plot(grid, np.array(fvals) - np.max(fvals))
        ax.set_title(index)
        plt.show()


def compute_profile_likelihood(params, options, df):

    crit_func = rp.get_crit_func(params, options, df)

    results = dict()
    for index in params[0:2].index:

        upper, lower = params.loc[index][["upper", "lower"]]
        grid = np.linspace(lower, upper, 20)

        fvals = list()
        for value in grid:
            params_copy = params.copy()
            params_copy.loc[index, "value"] = value
            fval = options["simulation_agents"] * crit_func(params_copy)
            fvals.append(fval)

            results[index] = fvals

    return results
