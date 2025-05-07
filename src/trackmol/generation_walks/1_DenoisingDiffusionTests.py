import random
import torch

print('PyTorch version:', torch.__version__)
from torch import nn

import pickle
import os

import torchvision

print('Torchvision version:', torchvision.__version__)
import torch.nn.functional as F

from tqdm import tqdm
from torchinfo import summary

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from data_utils import pad_trajectory
from models import *
import models

device = 'cuda' if torch.cuda.is_available() else 'cpu'
TYPES = ["CTRW", "fBM", "LW", "OU", "sBM"]


def plot_trajectory(ax, idx, traj, color="blue", time_step=None):
    if time_step is None:
        time_step = idx + 1
    traj = traj.squeeze()
    if traj.shape == torch.Size([2, 32]):
        traj = traj.permute(1, 0)
    assert traj.shape == torch.Size([32, 2])
    ax[idx].plot(traj[:, 0], traj[:, 1], marker='o', linestyle='-', markersize=5, color=color)
    ax[idx].set_title(f'Time step: {time_step}', fontsize=12, fontweight='bold')
    # ax[idx].set_xlim(-1, 1)  # Adjust based on expected range of motion
    # ax[idx].set_ylim(-1, 1)  # Adjust as needed
    ax[idx].grid(True)


"""## Visualise architecture"""

batch1 = torch.randn((64, 2, 32)).to(device)

print("linear variance sheduler", models.Linear_Variance_Scheduler)
ddpm = models.Linear_Variance_Scheduler(time_steps=1000, beta_start=0.0001, beta_end=0.02, device=device)

"""# Simple model"""

models = [item for item in os.listdir('.') if
          item.startswith("model") and item.endswith(".pth") and item != "model1.pth"]
print("Testing", models)

if os.path.exists("trajectories_test.pickle"):
    print("Loading test trajectories")
    with open("trajectories_test.pickle", "rb") as f:
        trajectories_training_dataset = pickle.load(f)
else:
    print("Test data not found.")
    training_trajectories = pd.read_csv('trajos_test.csv')
    training_trajectories = training_trajectories.groupby("n")
    training_trajectories = list(training_trajectories)
    training_trajectories = [tt[1] for tt in training_trajectories]
    for i in range(len(training_trajectories)):
        training_trajectories[i] = training_trajectories[i][["x", "y"]]
        training_trajectories[i].set_index(np.arange(len(training_trajectories[i])), inplace=True)
    cols = [f"h_{i}" for i in range(1, 17)]
    gratin_results = pd.read_csv("gratin_results_for_trajos_test.csv")
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

    with open("trajectories_test.pickle", "wb") as f:
        pickle.dump(trajectories_training_dataset, f)


def build_latent_reprojection_mse():
    model_to_total_loss = {}

    # Sample noise schedule (iss) once
    iss = torch.randn((1000, 1800, 2, 32)).to(device)
    # Sample the *same* initial noisy image
    initial_x = torch.randn((1800, 2, 32)).to(device)

    for model_name in models:
        model = torch.load(model_name, weights_only=False)
        print(f"Successfully loaded {model_name}.")
        model_to_total_loss[model_name] = 0
        os.makedirs("latent_projections_of_generated_data", exist_ok=True)
        with open(f"latent_projections_of_generated_data/reconstructed_trajs_{model_name}.csv", "w") as f:
            print("x,y,t,frame,n", file=f)
            trajectories = [traj for traj, _ in trajectories_training_dataset]
            trajectories = torch.cat(trajectories, dim=0)
            latent_representations = [latent_rep.unsqueeze(0) for _, latent_rep in trajectories_training_dataset]
            latent_representations = torch.cat(latent_representations, dim=0)
            print(trajectories.shape)

            _, collect = ddpm.ddpm_sampling(
                model=model,
                num_samples=1800,
                channels=2,
                traj_length=32,
                labels=latent_representations,
                x=initial_x,
                iss=iss
            )

            res = collect[-1].squeeze()
            for trajectory_index in range(res.shape[0]):
                for step_index in range(res.shape[2]):
                    print(
                        f"{res[trajectory_index][0][step_index].item()},{res[trajectory_index][1][step_index].item()},{(step_index + 1) / 100.:.4},{step_index + 1},{trajectory_index + 1}",
                        file=f)


def compute_latent_reprojection_mse():
    print("Loading actual projections.")
    actual_gratin_representations = [proj for _, proj in trajectories_training_dataset]
    print(f"\tdone: len = {len(actual_gratin_representations)}")
    for model_name in models:
        if ".pth" in model_name:
            model_name = model_name.split(".")[0]
        path = f"latent_projections_of_generated_data/reprojections/gratin_results_for_reconstructed_trajs_{model_name}.csv"
        if not os.path.exists(path):
            print(f"File {path} does not exist. Skipping.")
            continue
        results = pd.read_csv(path)
        results = results.groupby("n")
        results = [result[1] for result in results]
        results = [result[[f"h_{k}" for k in range(1, 17)]].to_numpy() for result in results]
        # compute mse for each trajectory
        mse_per_trajectory = []
        cossim_per_trajectory = []
        for actual, reconstructed in zip(actual_gratin_representations, results):
            reconstructed = torch.tensor(reconstructed).squeeze().to(device)
            actual = actual.squeeze().to(device)
            cossim = F.cosine_similarity(actual, reconstructed, dim=0)
            mse_loss = F.mse_loss(actual, reconstructed)
            cossim_per_trajectory.append(cossim.item())
            mse_per_trajectory.append(mse_loss.item())

        random_mse = []
        random_cossim = []
        for actual in actual_gratin_representations:
            actual = actual.squeeze().to(device)
            random_trajectory = random.choice(results)
            random_trajectory = torch.tensor(random_trajectory).squeeze().to(device)
            random_cossim.append(F.cosine_similarity(actual, random_trajectory, dim=0).item())
            random_mse.append(F.mse_loss(actual, random_trajectory).item())

        # compute mean mse
        mean_mse = np.mean(mse_per_trajectory)
        # print(f"Model {model_name}: Mean MSE = {mean_mse:.4f} vs. Random MSE = {np.mean(random_mse):.4f}")
        mean_cossim = np.mean(cossim_per_trajectory)
        # print(f"{model_name}: Mean Cosine Similarity = {mean_cossim:.4f} vs. Random Cosine Similarity = {np.mean(random_cossim):.4f}")
        print(
            f"{model_name} & {mean_cossim:.4f} & {np.mean(random_cossim):.4f} & {mean_mse:.4f} & {np.mean(random_mse):.4f} \\\\")


def replicate_experimental_data():
    original_trajectories = pd.read_csv("donnees_exp/raw.csv")
    cs = original_trajectories.groupby("n").count()
    idx = cs[cs.x == 32].index
    original_trajectories = original_trajectories.sort_values("n")
    original_trajectories = original_trajectories[original_trajectories.n.isin(idx)].groupby("n")
    original_trajectories = [torch.tensor(tr.sort_values("frame")[["x", "y"]].values).unsqueeze(0) for _, tr in
                             original_trajectories]
    original_trajectories = torch.concat(original_trajectories)
    original_trajectories = original_trajectories.float()

    embs = pd.read_csv("donnees_exp/gratin_results_for_truc.csv").sort_values("n")
    embs = embs[embs.n.isin(idx)].groupby("n")
    embs = [torch.tensor(emb[[f"h_{i}" for i in range(1, 17)]].values) for _, emb in embs]
    embs = torch.concat(embs)
    embs = embs.float()

    models = ["model2", "model3", "model4", "model5", "model6", "model1743763976"]
    os.makedirs(f"donnees_exp/replicated", exist_ok=True)
    for model_name in models:
        model = torch.load(f"{model_name}.pth", weights_only=False)
        _, collect = ddpm.ddpm_sampling(
            model=model,
            num_samples=len(embs),
            channels=2,
            traj_length=32,
            labels=embs,
        )
        res = collect[-1].squeeze()
        with open(f"donnees_exp/replicated/{model_name}.csv", "w") as f:
            print("x,y,t,frame,n", file=f)
            for trajectory_index in range(res.shape[0]):
                for step_index in range(res.shape[2]):
                    print(
                        f"{res[trajectory_index][0][step_index].item():.4},{res[trajectory_index][1][step_index].item():.4},{(step_index + 1) / 100.:.4},{step_index + 1},{trajectory_index + 1}",
                        file=f)


def experimental_compute_latent_reprojection_mse():
    original_trajectories = pd.read_csv("donnees_exp/raw.csv")
    cs = original_trajectories.groupby("n").count()
    idx = cs[cs.x == 32].index

    original_embeddings = pd.read_csv("donnees_exp/gratin_results_for_truc.csv")
    original_embeddings = original_embeddings[original_embeddings.n.isin(idx)].groupby("n")
    original_embeddings = [torch.tensor(emb[[f"h_{i}" for i in range(1, 17)]].values) for _, emb in original_embeddings]
    models = ["model2", "model3", "model4", "model5", "model6", "model1743763976"]

    for model_name in models:
        path = f"donnees_exp/embeddings_of_replicated/gratin_results_for_{model_name}.csv"
        if not os.path.exists(path):
            print(f"File {path} does not exist. Skipping.")
            continue
        embeddings_of_generated = pd.read_csv(path)
        embeddings_of_generated = torch.tensor(embeddings_of_generated[[f"h_{k}" for k in range(1, 17)]].values).to(
            device).float()
        # compute mse for each trajectory
        mse_per_trajectory = []
        cossim_per_trajectory = []
        for actual, reconstructed in zip(original_embeddings, embeddings_of_generated):
            reconstructed = reconstructed.squeeze().to(device)
            actual = actual.squeeze().to(device)
            cossim = F.cosine_similarity(actual, reconstructed, dim=0)
            mse_loss = F.mse_loss(actual, reconstructed)
            cossim_per_trajectory.append(cossim.item())
            mse_per_trajectory.append(mse_loss.item())

        random_mse = []
        random_cossim = []
        for actual in original_embeddings:
            actual = actual.squeeze().to(device)
            random_trajectory = random.choice(embeddings_of_generated)
            random_trajectory = random_trajectory.squeeze().to(device)
            random_cossim.append(F.cosine_similarity(actual, random_trajectory, dim=0).item())
            random_mse.append(F.mse_loss(actual, random_trajectory).item())

        # compute mean mse
        mean_mse = np.mean(mse_per_trajectory)
        # print(f"Model {model_name}: Mean MSE = {mean_mse:.4f} vs. Random MSE = {np.mean(random_mse):.4f}")
        mean_cossim = np.mean(cossim_per_trajectory)
        # print(f"{model_name}: Mean Cosine Similarity = {mean_cossim:.4f} vs. Random Cosine Similarity = {np.mean(random_cossim):.4f}")
        print(
            f"{model_name} & {mean_cossim:.4f} & {np.mean(random_cossim):.4f} & {mean_mse:.4f} & {np.mean(random_mse):.4f} \\\\")


def experimental_reprojection_dist():
    print("Coucou")
    models = ["model2", "model3", "model4", "model5", "model6", "model1743763976"]
    index = 1575
    # index =                   1792
    # index =                   2328

    experimental_trajectories = pd.read_csv("donnees_exp/raw.csv")
    __trajectory = experimental_trajectories[experimental_trajectories.n == index][["x", "y"]].values
    __trajectory = torch.tensor(__trajectory).to(device)
    embeddings = pd.read_csv("donnees_exp/gratin_results_for_truc.csv")
    latent_representation = embeddings[embeddings.n == index][[f"h_{i}" for i in range(1, 17)]].values
    latent_representation = torch.tensor(latent_representation).to(device)
    __trajectory = __trajectory.permute(1, 0).unsqueeze(0).float()
    latent_representation = latent_representation.squeeze().float()
    print("trajectory shape", __trajectory.shape)
    print("latent representation shape", latent_representation.shape)
    print(__trajectory.dtype, latent_representation.dtype)
    # Sample noise schedule (iss) once
    iss = torch.randn((1000, 1, 2, 32)).to(device)
    # Sample the *same* initial noisy image
    initial_x = torch.randn((1, 2, 32)).to(device)

    for model_idx, model_name in enumerate(models):
        print("Model", model_name)
        model = torch.load(f"{model_name}.pth", weights_only=False)

        # Copy the same initial x for this model
        x = initial_x.clone()
        _, collect = ddpm.ddpm_sampling(
            model=model,
            num_samples=1,
            channels=2,
            traj_length=32,
            labels=latent_representation,
            x=x,
            iss=iss,
            plot_frequency=200
        )

        fig, ax = plt.subplots(nrows=1, ncols=len(collect) + 1, figsize=(20, 4))
        for idx, trajectory in enumerate(collect):
            plot_trajectory(ax, idx, trajectory.to("cpu"), time_step=idx * 200)
        plot_trajectory(ax, len(collect), __trajectory.to("cpu"), color="orange")

        os.makedirs(f"donnees_exp/qualitative_test/{model_name}", exist_ok=True)
        plt.savefig(f"donnees_exp/qualitative_test/{model_name}/example_trajectories_{index}.png")
        plt.close()


def heatmap_compare_types_estimate_similar():
    all_generated_trajectories = {"BM": [], "CTRW": [], "fBM": [], "LW": [], "OU": [], "sBM": []}
    well_detected_trajectories = {"BM": [], "CTRW": [], "fBM": [], "LW": [], "OU": [], "sBM": []}

    for typ in all_generated_trajectories:
        all_trajectories = pd.read_csv(f"type_datasets/type_dataset/{typ}.csv").groupby("n")
        all_trajectories = list(all_trajectories)
        all_generated_trajectories[typ] = [traj[1] for traj in all_trajectories]
        all_embeddings = pd.read_csv(f"type_datasets/type_projs/gratin_results_for_{typ}.csv").groupby("n")
        all_embeddings = [emb[1] for emb in all_embeddings]
        well_detected_trajectories[typ] = [(traj, emb) for traj, emb in
                                           zip(all_generated_trajectories[typ], all_embeddings) if
                                           emb.best_model.values[0] == typ]
        print(f"For {typ} : {len(well_detected_trajectories[typ])} trajectories identified.")

    os.makedirs("type_datasets/generated/", exist_ok=True)
    for model_name in models:
        if ".pth" in model_name:
            model_name = model_name.split(".")[0]
        print("loading model", model_name)
        model = torch.load(f"{model_name}.pth", weights_only=False)
        print("\tmodel loaded")
        for typ, trajs in well_detected_trajectories.items():
            if not trajs:
                continue
            print("\n\n", trajs[0][1])
            print("\t\tGenerating trajectories for", typ)
            embs = torch.tensor([traj[1][[f"h_{i}" for i in range(1, 17)]].values[0] for traj in trajs]).to(
                device).float()
            print(f"\t\t\tembs len: {len(embs)}\tEmbeddings shape {embs.shape}\tEmbeddings dtype {embs.dtype}")
            _, collect = ddpm.ddpm_sampling(
                model=model,
                num_samples=len(trajs),
                channels=2,
                traj_length=32,
                labels=embs,
            )
            print("\t\t\tSampling done. Generated", len(collect), "trajectories.")
            res = collect[-1].squeeze()
            with open(f"type_datasets/generated/{model_name}_{typ}.csv", "w") as f:
                print("x,y,t,frame,n", file=f)
                for trajectory_index in range(res.shape[0]):
                    for step_index in range(res.shape[2]):
                        print(
                            f"{res[trajectory_index][0][step_index].item():.4},{res[trajectory_index][1][step_index].item():.4},{(step_index + 1) / 100.:.4},{step_index + 1},{trajectory_index + 1}",
                            file=f)
            print(f"\t\t\tWrote generated trajectories to type_datasets/generated/{model_name}_{typ}.csv.")


def visual_cmp():
    index = 42
    __trajectory, latent_representation = trajectories_training_dataset[index]
    latent_representation = torch.tensor(latent_representation).to(device)
    # Sample noise schedule (iss) once
    iss = torch.randn((1000, 1, 2, 32)).to(device)

    # Sample the *same* initial noisy image
    initial_x = torch.randn((1, 2, 32)).to(device)

    for model_idx, model_name in enumerate(models):
        model = torch.load(model_name, weights_only=False)

        # Copy the same initial x for this model
        x = initial_x.clone()

        _, collect = ddpm.ddpm_sampling(
            model=model,
            num_samples=1,
            channels=2,
            traj_length=32,
            labels=latent_representation,
            x=x,
            iss=iss,
            plot_frequency=200
        )

        fig, ax = plt.subplots(nrows=1, ncols=len(collect) + 1, figsize=(20, 4))
        for idx, trajectory in enumerate(collect):
            plot_trajectory(ax, idx, trajectory.to("cpu"), time_step=idx * 200)
        plot_trajectory(ax, len(collect), __trajectory.to("cpu"), color="orange")

        plt.savefig(f"""tests/small/{model_name.split(".")[0]}_example_trajectories_{index}.png""")
        plt.close()


def build_heatmap():
    models = ["model2", "model3", "model4", "model5", "model6", "model1743763976"]
    model_name_to_verbose = {
        "model2": "Convolution",
        "model3": "Unet(256)",
        "model4": "Unet(128)",
        "model5": "Unet(16)",
        "model6": "Unet(1024)",
        "model1743763976": "Attention simple"
    }
    hm = np.zeros((len(models), len(TYPES)))
    true_positives = {}
    false_positives = {(m, t): 0 for m in models for t in TYPES}
    false_negatives = {(m, t): 0 for m in models for t in TYPES}
    for model_name in models:
        for typ in TYPES:
            generated_data_embeddings = pd.read_csv(
                f"type_datasets/gratin_results_for_generated/gratin_results_for_{model_name}_{typ}.csv")
            correctly_identified_as_typ = generated_data_embeddings[
                generated_data_embeddings.best_model == typ].best_model.count()
            true_positives[(model_name, typ)] = correctly_identified_as_typ
            false_negatives[(model_name, typ)] = generated_data_embeddings[
                generated_data_embeddings.best_model != typ].best_model.count()
            for best_model in generated_data_embeddings[generated_data_embeddings.best_model != typ].best_model:
                if best_model != typ:
                    false_positives[(model_name, best_model)] += 1

    # plot f score heatmap
    f_score = np.zeros((len(models), len(TYPES)))
    for i, model_name in enumerate(models):
        for j, typ in enumerate(TYPES):
            tp = true_positives[(model_name, typ)]
            fn = false_negatives[(model_name, typ)]
            fp = false_positives[(model_name, typ)]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f_score[i][j] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(f_score, cmap='coolwarm')
    plt.colorbar(cax)
    ax.set_xticks(np.arange(len(TYPES)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(TYPES)
    ax.set_yticklabels([model_name_to_verbose[m] for m in models])
    plt.xlabel("Types")
    plt.ylabel("Models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("heatmap_fscore.png")


def build_confusin_matrix(model):
    cm = np.zeros((len(TYPES), len(TYPES)))
    for typ in TYPES:
        generated_data_embeddings = pd.read_csv(
            f"type_datasets/gratin_results_for_generated/gratin_results_for_{model}_{typ}.csv")
        print(len(generated_data_embeddings))
        for i, row in random.sample(sorted(generated_data_embeddings.iterrows()), k=62):
            cm[TYPES.index(row.best_model)][TYPES.index(typ)] += 1

    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(cm, cmap='coolwarm')
    plt.colorbar(cax)
    ax.set_xticks(np.arange(len(TYPES)))
    ax.set_yticks(np.arange(len(TYPES)))
    ax.set_xticklabels(TYPES)
    ax.set_yticklabels(TYPES)
    plt.xlabel("Vérité")
    plt.ylabel("Prédiction")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{model}.png")


if __name__ == '__main__':
    # compute_latent_reprojection_mse()
    # experimental_compute_latent_reprojection_mse()
    # replicate_experimental_data()
    # heatmap_compare_types_estimate_similar()
    # build_heatmap()
    # build_confusin_matrix("model5")
    visual_cmp()
    # experimental_reprojection_dist()
