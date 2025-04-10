import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


def main():
    folders_path = [
        "/workspace/isaaclab/logs/rsl_rl/jetbot_direct/2025-04-09_12-53-32_rsl-rl_TrackVelocities_Jetbot_r-0_seed-42/",
        "/workspace/isaaclab/logs/rsl_rl/jetbot_direct/2025-04-09_12-53-32_rsl-rl_TrackVelocities_Jetbot_r-0_seed-42/",
        "/workspace/isaaclab/logs/rsl_rl/jetbot_direct/2025-04-09_12-53-32_rsl-rl_TrackVelocities_Jetbot_r-0_seed-42/",
        "/workspace/isaaclab/logs/rsl_rl/jetbot_direct/2025-04-09_12-53-32_rsl-rl_TrackVelocities_Jetbot_r-0_seed-42/",
    ]

    dfs = []
    labels = []
    for folder_path in folders_path:
        experiment_name = glob.glob(os.path.join(folder_path, "metrics", "*_metrics.csv"))[0]
        file_path = os.path.join(folder_path, "metrics", experiment_name)
        df = pd.read_csv(file_path)
        dfs.append(df)
        labels.append(experiment_name.split("/")[-1])

    boxplot(dfs, labels, "TrackVelocities/linear_velocity_error")

    breakpoint()


def boxplot(dfs: list, labels: list, key_to_plot: str):
    data_to_plot = [df[key_to_plot] for df in dfs]

    plt.figure(figsize=(10, 6))

    plt.boxplot(
        data_to_plot,
        labels=labels,
        patch_artist=True,
        boxprops=dict(facecolor='skyblue'),
        medianprops=dict(color='black'),
        flierprops=dict(marker='o', markerfacecolor='red', markersize=5),
        positions=list(range(1, len(dfs) + 1)), # Dynamic positions
        widths=0.6,
    )

    plt.title(f"Boxplots of {key_to_plot}")
    plt.ylabel(key_to_plot)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("source/test_2.png")
    plt.close()


        


    # df_1 = pd.read_csv('/workspace/isaaclab/metrics/rsl_rl/Jetbot_GoToPosition_2025-04-08-21-16-38_metrics.csv')
    # df_2 = pd.read_csv('/workspace/isaaclab/metrics/rsl_rl/Jetbot_GoToPosition_2025-04-08-21-32-20_metrics.csv')
    # df_3 = pd.read_csv('/workspace/isaaclab/metrics/rsl_rl/Jetbot_GoToPosition_2025-04-08-21-35-12_metrics.csv')
    # df_4 = pd.read_csv('/workspace/isaaclab/metrics/rsl_rl/Jetbot_GoToPosition_2025-04-08-21-38-31_metrics.csv')

    # plt.figure(figsize=(10, 6))

    # key_to_plot = "GoToPosition/trajectory_efficiency"
    # labels = ['21-16-38', '21-32-20', '21-35-12', '21-38-31']

    # # Create boxplots using matplotlib's boxplot function
    # plt.boxplot(
    #     [df_1[key_to_plot], df_2[key_to_plot], df_3[key_to_plot], df_4[key_to_plot]],
    #     labels=labels, patch_artist=True,
    #     boxprops=dict(facecolor='skyblue'),
    #     medianprops=dict(color='black'),
    #     flierprops=dict(marker='o', markerfacecolor='red', markersize=5),
    #     positions = [1,2,3,4], 
    #     widths = 0.6
    # )

    # plt.title("Boxplots of Trajectory Efficiency")
    # plt.ylabel("Trajectory Efficiency")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("source/test.png")


if __name__ == "__main__":
    main()
    