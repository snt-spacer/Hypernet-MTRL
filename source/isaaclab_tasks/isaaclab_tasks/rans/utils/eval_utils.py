import torch

class EvalMetrics:
    def __init__(self, env):
        self._env = env

    def calculate_metrics(self, data: dict)->None:

        trajectories, trajectories_masks = self.split_and_pad_trajectories(data)

        print("[INFO] Evaluating metrics...")

    def split_and_pad_trajectories(self, data: dict) -> tuple[dict[str, torch.Tensor], torch.Tensor]:

        dones_tensor = torch.stack(data["dones"]) # Shape [timesteps, num_envs]
        num_transitions_per_env = dones_tensor.shape[0]
        num_envs = dones_tensor.shape[1]
        device = dones_tensor.device

        dones_calc = dones_tensor.clone()
        dones_calc[-1] = 1 # Ensure final step terminates all trajectories

        flat_dones = dones_calc.transpose(1, 0).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64),
             flat_dones.nonzero(as_tuple=False)[:, 0])
        )

        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        max_traj_len = trajectory_lengths.max().item()

        traj_range = torch.arange(max_traj_len, device=device)
        trajectories_mask = traj_range[None, :] < trajectory_lengths[:, None]
        
        extracted_trajectories = {}
        trajectories_masks_dict = {}

        for key, list_tensor in data.items():
            if key == 'dones':
                continue
                
            if not list_tensor:
                extracted_trajectories[key] = torch.empty(0, device=device)
                trajectories_masks_dict[key] = torch.empty(0, dtype=torch.bool, device=device)
                continue


            tensor_log = torch.stack(list_tensor)
            data_shape = tensor_log.shape[2:] # Get original feature dimensions

            # Shape becomes [total_steps, feat_dim] where total_steps = num_envs * timesteps
            flat_tensor = tensor_log.transpose(1, 0).reshape(num_envs * num_transitions_per_env, *data_shape)

            # Split the flattened tensor into a list of tensors based on trajectory lengths
            list_of_trajectories = list(flat_tensor.split(trajectory_lengths.tolist(), dim=0))

            # Pad the list of trajectory tensors [num_trajectories, max_traj_len, feat_dim]
            padded_tensor = torch.nn.utils.rnn.pad_sequence(
                list_of_trajectories,
                batch_first=True,
                padding_value=0
            )

            extracted_trajectories[key] = padded_tensor

        return extracted_trajectories, trajectories_mask