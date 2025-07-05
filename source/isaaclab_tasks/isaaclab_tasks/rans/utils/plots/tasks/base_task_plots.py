import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from matplotlib.ticker import MaxNLocator
import random
from matplotlib.collections import LineCollection
import matplotlib as mpl

class AutoRegister:
    def __init_subclass__(cls, **kwargs):
        """Ensure each subclass gets its own independent registry."""
        super().__init_subclass__(**kwargs)

        # Unique for each subclass + inherit from parent class
        cls._registry = getattr(super(cls, cls), '_registry', {}).copy()

        for name, value in cls.__dict__.items():
            # If an attribute is a function and has our marker, register it.
            if callable(value) and getattr(value, '_auto_register', False):
                cls._registry[name] = value

    @staticmethod
    def register(func: callable) -> callable:
        """Decorator that simply marks a function so that __init_subclass__
        knows it should be placed in the registry.
        """
        func._auto_register = True
        return func

    @classmethod
    def get_registered_methods(cls) -> dict[str, callable]:
        """Retrieve registered methods."""
        return cls._registry


class BaseTaskPlots(AutoRegister):
    def __init__(self, dfs: dict, trajectories_dfs: dict, labels: dict, env_info: dict, folder_path: list, plot_cfg:dict) -> None:
        self._dfs = dfs
        self._trajectories_dfs = trajectories_dfs
        self._labels = labels
        self._env_info = env_info
        self._save_plots_folder_path = folder_path
        self._plot_cfg = plot_cfg

        dfs_to_concat = []
        trajectory_offset = 0

        for group_key, group_dfs in self._trajectories_dfs.items():
            for df in group_dfs:
                df = df.copy()
                df['trajectory'] += trajectory_offset
                dfs_to_concat.append(df)
                max_traj = df['trajectory'].max()
                trajectory_offset = max_traj + 1

        self.trajectories_to_plot = pd.concat(dfs_to_concat, ignore_index=True)

        self.ALPHA_VALUE = 0.8

        trajectory_names = self.trajectories_to_plot['trajectory'].unique()
        self.trajectory_color_map_hex = {name: "#%06x" % random.randint(0, 0xFFFFFF) for name in trajectory_names}


    def plot(self):
        raise NotImplementedError("Subclasses should implement this method.")
    

    def boxplot(self, key_to_plot: str):
        key_name, units = key_to_plot.split(".")

        fig, ax = plt.subplots(figsize=(14, 6))

        data_to_plot = []
        label_names = []

        
        colors_indx = []
        for group_key, group_dfs in self._dfs.items():
            # Concatenate all values of the key across the group's dataframes
            # for group_idx, df in enumerate(group_dfs):
            #     print("#" * 20)
            #     print(f"Group {group_idx}: {df.columns}")

            try:
                group_values = pd.concat([df[key_to_plot] for df in group_dfs], ignore_index=True)
            except KeyError as e:
                print(f"KeyError: {e}. The key '{key_to_plot}' does not exist in the DataFrames.")
                print(self.__class__.__name__)
                return
            
            colors_indx.append(self._plot_cfg['runs_names'].index(group_key.split("_")[-1]))
            
            data_to_plot.append(group_values)
            label_names.append(group_key.split("_")[-1])

        box = ax.boxplot(
            data_to_plot,
            labels=label_names,
            patch_artist=True,
            boxprops=dict(facecolor='skyblue'),
            medianprops=dict(color='black'),
            flierprops=dict(marker='o', markerfacecolor='red', markersize=5),
            showfliers=False,
            widths=0.6,
        )

        if self._plot_cfg["zoom_in"]:
            all_values = pd.concat(data_to_plot)
            ax.set_ylim(top=all_values.quantile(0.95) )

        # Set the color of the boxes
        colors = [self._plot_cfg["box_colors"][index] for index in colors_indx]
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        y_label, units = key_to_plot.split(".")
        y_label = y_label.replace("_", " ").capitalize()
        ax.set_title(f"{y_label}")
        ax.set_ylabel(f"{y_label} ({units})")
        ax.set_xticks(range(1, len(label_names) + 1))
        ax.set_xticklabels(label_names, rotation=20, ha='right')
        ax.grid(True)
        plt.tight_layout()
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_{key_name}.svg")
        plt.savefig(save_path)
        plt.close()

    def dotplot(self, key_to_plot: str):
        key_name, units = key_to_plot.split(".")

        fig, ax = plt.subplots(figsize=(14, 6))

        data_to_plot = []
        label_names = []

        colors_indx = []
        for group_key, group_dfs in self._dfs.items():
            try:
                group_values = pd.concat([df[key_to_plot] for df in group_dfs], ignore_index=True)
            except KeyError as e:
                print(f"KeyError: {e}. The key '{key_to_plot}' does not exist in the DataFrames.")
                print(self.__class__.__name__)
                return
            
            colors_indx.append(self._plot_cfg['runs_names'].index(group_key.split("_")[-1]))
            
            data_to_plot.append(group_values)
            label_names.append(group_key.split("_")[-1])

        # Create a dot plot
        for i, (values, label) in enumerate(zip(data_to_plot, label_names)):
            ax.plot([i + 1] * len(values), values, 'o', color=self._plot_cfg["box_colors"][colors_indx[i]], markersize=5)

        if self._plot_cfg["zoom_in"]:
            all_values = pd.concat(data_to_plot)
            ax.set_ylim(top=all_values.quantile(0.95) )

        y_label, units = key_to_plot.split(".")
        y_label = y_label.replace("_", " ").capitalize()
        ax.set_title(f"{y_label}")
        ax.set_ylabel(f"{y_label} ({units})")
        ax.set_xticks(range(1, len(label_names) + 1))
        ax.set_xticklabels(label_names, rotation=20, ha='right')
        ax.grid(True)
        plt.tight_layout()
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_{key_name}.svg")
        plt.savefig(save_path)
        plt.close()

    def plot_mean_sd(self, data_df, y_col, fname, ylabel=None):
        grouped = data_df.groupby('step')[y_col]
        mean = grouped.mean()
        std = grouped.std()
        steps = mean.index
        plt.figure(figsize=(8,6))
        plt.plot(steps, mean, label='Mean', color='b')
        plt.fill_between(steps, mean-std, mean+std, color='b', alpha=0.2, label='±1 SD')
        plt.title(f'{y_col.replace("_", " ").title()} Mean ± SD Over Time')
        plt.xlabel('Step')
        plt.ylabel(ylabel if ylabel else y_col)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(fname)
        plt.close()

    def plot_trajectories(self, ax, data_df, x_col, y_col, title, xlabel, ylabel, x_lim=None, y_lim=None, add_target_lines=False):
        """
        Plots individual trajectories with distinct random hex colors using a global trajectory_color_map_hex.
        """
        for trajectory_name, group in data_df.groupby('trajectory'):
            # Get the pre-assigned random hex color for this trajectory
            hex_color = self.trajectory_color_map_hex[trajectory_name]
            ax.plot(group[x_col], group[y_col], color=hex_color, alpha=self.ALPHA_VALUE, label=f'Trajectory {trajectory_name}')

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if add_target_lines:
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax.axvline(0, color='gray', linestyle='--', linewidth=1)
            ax.set_aspect('equal', adjustable='box') # Only for normalized XY plot

        if x_lim:
            ax.set_xlim(x_lim[0], x_lim[1])
        if y_lim:
            ax.set_ylim(y_lim[0], y_lim[1])

    def plot_trajectories_with_gradient(self, ax, data_df, x_col, y_col, step_col='step', cmap_name='viridis'):
        for trajectory_name, group in data_df.groupby('trajectory'):
            group = group.sort_values(step_col)
            x = group[x_col].values
            y = group[y_col].values
            steps = group[step_col].values

            # Normalize steps to [0, 1] for colormap
            norm = mpl.colors.Normalize(vmin=steps.min(), vmax=steps.max())
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(
                segments,
                cmap=cmap_name,
                norm=norm,
                array=steps[:-1],  # color by step
                linewidth=2,
                alpha=1.0
            )
            ax.add_collection(lc)
            ax.autoscale_view()

        sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Step')

    def plot_xy_trajectories_0_centered(self):
        xy_df = self.trajectories_to_plot.copy()
        xy_df['tx'] = xy_df.groupby('trajectory')['target_position_x'].transform('first')
        xy_df['ty'] = xy_df.groupby('trajectory')['target_position_y'].transform('first')
        xy_df['norm_position_x'] = xy_df['position_x'] - xy_df['tx']
        xy_df['norm_position_y'] = xy_df['position_y'] - xy_df['ty']
        xy_df = xy_df.drop(columns=['tx', 'ty'])

        fig, ax = plt.subplots(figsize=(8,8))
        self.plot_trajectories_with_gradient(ax, xy_df, 'norm_position_x', 'norm_position_y', step_col='step', cmap_name='viridis')
        ax.set_title('XY Trajectory (Target at 0,0)')
        ax.set_xlabel('Normalized Position X')
        ax.set_ylabel('Normalized Position Y')
        # ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        # ax.axvline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_xy-trajectory-normalized.svg")
        plt.savefig(save_path)
        plt.close(fig)

    def plot_position_distance_over_time(self):
        fig2, ax2 = plt.subplots(figsize=(8,6))
        self.plot_trajectories(
            ax2, self.trajectories_to_plot, 'step', 'position_distance',
            'Position Distance to Target Over Time', 'Step', 'Distance'
        )
        plt.tight_layout()
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_position-distance-over-time.svg")
        plt.savefig(save_path)
        plt.close(fig2)

        # Position distance mean ± sd
        dist_col = 'position_distance'
        self.plot_mean_sd(self.trajectories_to_plot, dist_col, os.path.join(self._save_plots_folder_path, f"{self.task_name}_position-distance-over-time-mean-sd.svg"), ylabel='Distance')

    def plot_linear_velocity_over_time(self):
        fig4, axs4 = plt.subplots(2, 1, figsize=(10,8), sharex=True)
        self.plot_trajectories(
            axs4[0], self.trajectories_to_plot, 'step', 'linear_velocity_x',
            'Linear Velocity X Over Time', 'Step', 'Linear Velocity X'
        )
        self.plot_trajectories(
            axs4[1], self.trajectories_to_plot, 'step', 'linear_velocity_y',
            'Linear Velocity Y Over Time', 'Step', 'Linear Velocity Y'
        )

        axs4[-1].set_xlabel('Step') # Set x-label only on the last subplot for shared x-axis
        plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust rect for legend
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_linear-velocities-over-time.svg")
        plt.savefig(save_path)
        plt.close(fig4)

        # Combined mean±SD plot for linear velocities (x and y)
        fig7, axs7 = plt.subplots(2, 1, figsize=(10,8), sharex=True)
        for i, col in enumerate(['linear_velocity_x', 'linear_velocity_y']):
            grouped = self.trajectories_to_plot.groupby('step')[col]
            mean = grouped.mean()
            std = grouped.std()
            steps = mean.index
            axs7[i].plot(steps, mean, label='Mean', color='b')
            axs7[i].fill_between(steps, mean-std, mean+std, color='b', alpha=0.2, label='±1 SD')
            axs7[i].set_title(f'{col.replace("_", " ").title()} Mean ± SD Over Time')
            axs7[i].set_ylabel(col)
            axs7[i].legend()
        axs7[-1].set_xlabel('Step')
        plt.tight_layout()
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_linear-velocities-over-time-mean-sd.svg")
        plt.savefig(save_path)
        plt.close(fig7)

    # GoToPosition, GoToPose
    def plot_angular_velocity_over_time(self):
        fig6, ax6 = plt.subplots(figsize=(8,6))
        self.plot_trajectories(
            ax6, self.trajectories_to_plot, 'step', 'angular_velocity_z',
            'Angular Velocity Z Over Time', 'Step', 'Angular Velocity Z'
        )
        plt.tight_layout()
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_angular-velocity-over-time.svg")
        plt.savefig(save_path)
        plt.close(fig6)

        self.plot_mean_sd(self.trajectories_to_plot, 'angular_velocity_z', os.path.join(self._save_plots_folder_path, f"{self.task_name}_angular-velocity-over-time-mean-sd.svg"), ylabel='Angular Velocity Z')

    def plot_actions_over_time(self):
        action_cols = [col for col in self.trajectories_to_plot.columns if col.startswith('actions_')]
        fig5, axs5 = plt.subplots(len(action_cols), 1, figsize=(10, 2*len(action_cols)), sharex=True)
        if len(action_cols) == 1:
            axs5 = [axs5]

        for i, col in enumerate(action_cols):
            self.plot_trajectories(
                axs5[i], self.trajectories_to_plot, 'step', col,
                f'{col} Over Time', 'Step', col
            )
        axs5[-1].set_xlabel('Step')
        plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust rect for legend
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_actions-over-time.svg")
        plt.savefig(save_path)
        plt.close(fig5)

    # ----------------------------- TrackVelocitiesPlots -----------------------------
    def plot_linear_velocity_error_over_time(self):
        fig4, axs4 = plt.subplots(2, 1, figsize=(10,8), sharex=True)
        self.plot_trajectories(
            axs4[0], self.trajectories_to_plot, 'step', 'error_linear_velocity',
            'Error Linear Velocity X Over Time', 'Step', 'Error Linear Velocity X'
        )
        self.plot_trajectories(
            axs4[1], self.trajectories_to_plot, 'step', 'error_lateral_velocity',
            'Error Linear Velocity Y Over Time', 'Step', 'Error Linear Velocity Y'
        )

        axs4[-1].set_xlabel('Step') # Set x-label only on the last subplot for shared x-axis
        plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust rect for legend
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_linear-velocities-error-over-time.svg")
        plt.savefig(save_path)
        plt.close(fig4)

    def plot_angular_velocity_error_over_time(self):
        fig6, ax6 = plt.subplots(figsize=(8,6))
        self.plot_trajectories(
            ax6, self.trajectories_to_plot, 'step', 'error_angular_velocity',
            'Error Angular Velocity Z Over Time', 'Step', 'Error Angular Velocity Z'
        )
        plt.tight_layout()
        save_path = os.path.join(self._save_plots_folder_path, f"{self.task_name}_angular-velocity-error-over-time.svg")
        plt.savefig(save_path)
        plt.close(fig6)

        