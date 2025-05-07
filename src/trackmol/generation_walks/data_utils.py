import torch


def pad_trajectory(trajectory, target_length):
    num_rows_to_pad = target_length - trajectory.shape[0]
    last_row = trajectory[-1].unsqueeze(0)  # Shape (1, 2)
    padding = last_row.repeat(num_rows_to_pad, 1)
    zeros = torch.zeros_like(padding)
    res = torch.cat([trajectory, zeros], dim=0)
    assert res.shape == (target_length, 2)
    return res


