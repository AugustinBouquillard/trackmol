import datetime
from sys import stderr

import torch

from models import DenoiseTrajectoryNet_pooling

print('PyTorch version:', torch.__version__)
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import pickle
import os
import time

from data_utils import pad_trajectory
from models import *

import models

import torchvision

print('Torchvision version:', torchvision.__version__)
from torchinfo import summary

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

print('GPU name:', torch.cuda.get_device_name(), '\n')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device is:', device, '\n')
print('Total number of GPUs:', torch.cuda.device_count())

modeln = int(time.time())


def plot_trajectory(ax, idx, traj):
    traj = traj.squeeze()
    if traj.shape == torch.Size([2, 32]):
        traj = traj.permute(1, 0)
    assert traj.shape == torch.Size([32, 2])
    ax[idx].plot(traj[:, 0], traj[:, 1], marker='o', linestyle='-', markersize=5)
    ax[idx].set_title(f'Time step: {idx + 1}', fontsize=12, fontweight='bold')
    # ax[idx].set_xlim(-1, 1)  # Adjust based on expected range of motion
    # ax[idx].set_ylim(-1, 1)  # Adjust as needed
    ax[idx].grid(True)


if os.path.exists("trajectories_training_dataset.pickle") and os.path.exists("trajectories_test_dataset.pickle"):
    print("Loading datasets")
    with open("trajectories_training_dataset.pickle", "rb") as f:
        trajectories_training_dataset = pickle.load(f)
    with open("trajectories_test_dataset.pickle", "rb") as f:
        trajectories_test_dataset = pickle.load(f)
else:
    print("Datasets not found.")
    training_trajectories = pd.read_csv('trajos2.csv')
    training_trajectories = training_trajectories.groupby("n")
    training_trajectories = list(training_trajectories)
    training_trajectories = [tt[1] for tt in training_trajectories]
    for i in range(len(training_trajectories)):
        training_trajectories[i] = training_trajectories[i][["x", "y"]]
        training_trajectories[i].set_index(np.arange(len(training_trajectories[i])), inplace=True)
    cols = [f"h_{i}" for i in range(1, 17)]
    gratin_results = pd.read_csv("gratin_results_for_trajos2.csv")
    gratin_results = gratin_results[cols]
    training_trajectories_padded = [
        pad_trajectory(torch.tensor(training_trajectories[i].values), 32).unsqueeze(0)
        for i in range(len(training_trajectories))
    ]
    for i, traj in enumerate(training_trajectories_padded):
        training_trajectories_padded[i] = traj.permute(0, 2, 1)
    trajectories_training_dataset = [(training_trajectories_padded[i].to(torch.float32),
                                      torch.tensor(gratin_results.iloc[i].values.astype(np.float32))) for i in
                                     range(len(training_trajectories))]

    TRAINING_SET_SIZE = (8 * len(trajectories_training_dataset)) // 10

    trajectories_training_dataset, trajectories_test_dataset = trajectories_training_dataset[
                                                               :TRAINING_SET_SIZE], trajectories_training_dataset[
                                                                                    TRAINING_SET_SIZE:]
    print(len(trajectories_training_dataset), len(trajectories_test_dataset))

    with open("trajectories_training_dataset.pickle", "wb") as f:
        pickle.dump(trajectories_training_dataset, f)
    with open("trajectories_test_dataset.pickle", "wb") as f:
        pickle.dump(trajectories_test_dataset, f)

trajectories_training_dataloader = DataLoader(trajectories_training_dataset, batch_size=64, shuffle=True,
                                              drop_last=True)
trajectories_test_dataloader = DataLoader(trajectories_test_dataset, batch_size=64, shuffle=False, drop_last=True)


#ddpm = Linear_Variance_Scheduler(time_steps=1000, beta_start=0.0001, beta_end=0.02, device=device)
# model = Conditional_UNet(t_emb_dim=1024, device=device).to(device)  #
#criterion = nn.MSELoss()
#optimizer = optim.Adam(model.parameters(), lr=0.01)

"""# Training"""

model = models.Conditional_UNet_cross_attention(t_emb_dim=1024, device=device).to(device)
ddpm = Linear_Variance_Scheduler(time_steps=1000, beta_start=0.0001, beta_end=0.02, device=device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("TRAINING", file=stderr)
print("TRAINING")
print("Model:", type(model))

torch.manual_seed(1111)
torch.random.manual_seed(1111)
torch.cuda.manual_seed(1111)
torch.cuda.manual_seed_all(1111)
np.random.seed(1111)

n_epochs = 200
training_loss, test_loss = [], []

def training_extra_layers():
    start = time.time()
    for epoch in range(n_epochs):
        print(f"Beginning expoch {epoch}", file=stderr)
        training_losses, test_losses = [], []

        for data, labels in trajectories_training_dataloader:  #
            model.train()
            labels = labels.to(device)
            data = data.to(device)

            t = torch.randint(low=0, high=1000, size=(data.shape[0],)).to(device)
            noise = torch.randn_like(data)
            xt = ddpm.diffusion_process(x=data, noise=noise, t=t)
            pred_noise = model(xt, t, labels)
            trng_batch_loss = criterion(noise.squeeze(), pred_noise)
            optimizer.zero_grad()
            trng_batch_loss.backward()
            optimizer.step()
            training_losses.append(trng_batch_loss.item())
        training_per_epoch_loss = np.array(training_losses).mean()

        with torch.inference_mode():
            for data, labels in trajectories_test_dataloader:  #
                model.eval()
                data = data.to(device)
                #
                labels = labels.to(device)
                t = torch.randint(low=0, high=1000, size=(data.shape[0],)).to(device)
                noise = torch.randn_like(data)
                xt = ddpm.diffusion_process(x=data, noise=noise, t=t)
                pred_noise = model(xt, t, labels)  #
                noise = noise.squeeze()
                tst_batch_loss = criterion(noise, pred_noise)
                test_losses.append(tst_batch_loss.item())
            test_per_epoch_loss = np.array(test_losses).mean()

        training_loss.append(training_per_epoch_loss)
        test_loss.append(test_per_epoch_loss)

        print(f'Epoch: {epoch + 1}/{n_epochs}\t| Training loss: {training_per_epoch_loss:.4f} |    ', end='')
        print(f'Test loss: {test_per_epoch_loss:.4f}')
    end = time.time()
    print(f"Finished training in {end - start}s")

if False:
    # convert model to ONNX
    print("Exporting model to ONNX format")
    dummy_input = torch.randn(1, 2, 32).to(device)
    dummy_t = torch.randint(low=0, high=1000, size=(1,)).to(device)
    dummy_labels = torch.randn(1, 16).to(device)
    dummy_output = model(dummy_input, dummy_t, dummy_labels)
    torch.onnx.export(model, (dummy_input, dummy_t, dummy_labels), f"{modeln}.onnx",
                      input_names=["input", "t", "labels"],
                      output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "t": {0: "batch_size"}, "labels": {0: "batch_size"},
                                    "output": {0: "batch_size"}})
    print(f"Model exported to ONNX format as {modeln}.onnx.")

ddpm = Linear_Variance_Scheduler(time_steps=1000, beta_start=0.0001, beta_end=0.02, device=device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

"""## Visualise architecture"""

print(device)
batch1 = torch.randn((64, 2, 32)).to(device)
summary(model, input_size=[(64, 2, 32), (64,), (64, 16)],
        dtypes=[torch.float, torch.float, torch.float], device=device)

print("Model printed", file=stderr)

start = time.time()
for epoch in range(n_epochs):
    print(f"Beginning epoch {epoch}", file=stderr)
    training_losses, test_losses = [], []

    for data, labels in trajectories_training_dataloader:  #
        model.train()
        labels = labels.to(device)
        data = data.to(device)

        t = torch.randint(low=0, high=1000, size=(data.shape[0],)).to(device)
        noise = torch.randn_like(data)
        xt = ddpm.diffusion_process(x=data, noise=noise, t=t)
        pred_noise = model(xt, t, labels)
        trng_batch_loss = criterion(noise.squeeze(), pred_noise)
        optimizer.zero_grad()
        trng_batch_loss.backward()
        optimizer.step()
        training_losses.append(trng_batch_loss.item())
    training_per_epoch_loss = np.array(training_losses).mean()

    with torch.inference_mode():
        for data, labels in trajectories_test_dataloader:  #
            model.eval()
            data = data.to(device)
            #
            labels = labels.to(device)
            t = torch.randint(low=0, high=1000, size=(data.shape[0],)).to(device)
            noise = torch.randn_like(data)
            xt = ddpm.diffusion_process(x=data, noise=noise, t=t)
            pred_noise = model(xt, t, labels)  #
            noise = noise.squeeze()
            tst_batch_loss = criterion(noise, pred_noise)
            test_losses.append(tst_batch_loss.item())
        test_per_epoch_loss = np.array(test_losses).mean()

    training_loss.append(training_per_epoch_loss)
    test_loss.append(test_per_epoch_loss)

    print(f'Epoch: {epoch + 1}/{n_epochs}\t| Training loss: {training_per_epoch_loss:.4f} |    ', end='')
    print(f'Test loss: {test_per_epoch_loss:.4f}')
end = time.time()
print(f"Finished training in {end - start}s")

"""# Plot loss"""
print("PLOT LOSS")

os.makedirs(f"medias/{modeln}/", exist_ok=True)

N = np.arange(n_epochs) + 1

plt.figure(figsize=(10, 4))
plt.plot(N, training_loss, 'm-', linewidth=3, label='Training loss')
plt.plot(N, test_loss, 'b-.', linewidth=3, label='Test loss')
plt.title('Loss curve', fontsize=25)
plt.xlabel('No. of epochs', fontsize=18)
plt.ylabel('Losses', fontsize=18)
plt.xticks(N, fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize=15)
plt.savefig(f"medias/{modeln}/loss.png")
plt.close()

"""# Show example of reconstructed trajectory"""

indices = np.random.randint(0, len(trajectories_test_dataset), 10)
for index in indices:
    trajectory, latent_representation = trajectories_test_dataset[index]
    latent_representation = torch.tensor(latent_representation).to(device)
    _, collect = ddpm.ddpm_sampling(model=model, num_samples=1, channels=2, traj_length=32,
                                    labels=latent_representation)
    fig, ax = plt.subplots(nrows=1, ncols=11, figsize=(40, 4))
    for idx, trajectory in enumerate(collect):
        plot_trajectory(ax, idx, trajectory.to("cpu"))

    plt.savefig(f"medias/{modeln}/example_trajectories_{index}.png")
    plt.close()

print(f"Saving to model{modeln}")
torch.save(model, f"model{modeln}.pth")
