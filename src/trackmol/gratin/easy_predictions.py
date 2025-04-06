#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:50:05 2024

@author: augustinbouquillard
"""

# caution: path[0] is reserved for script path (or '' in REPL)
"""from standard import load_model, plot_demo

model, encoder = load_model(export_path="tracktor2D")
"""

from trackmol.gratin.standard import load_model, plot_demo
from trackmol.gratin.standard import get_predictions, trajectory_is_valid
import pandas as pd
import numpy as np
import os

def easy_predictions(filename,plot=False):

    #trajectories = pd.read_csv('n2_+myoIb_1_trackedPar.csv')
    #trajectories = pd.read_csv('Video6_test.csv')
    if os.path.isdir(filename):
        for file in os.listdir(filename):
            file_name = str(filename)+"/"+os.fsdecode(file)

            trajectories = pd.read_csv(file_name)
            model, encoder = load_model(export_path="tracktor2D")

            predictions = get_predictions(model, encoder, trajectories)

            if plot:
                plot_demo(
                model,
                encoder,
                length_range = (7, 55), # these values can differ from those used during training
                noise_range = (0.015, 0.05)
                )

            trajectories_indices = [(n, t[["x", "y"]].values, t["t"].values) for n, t in trajectories.sort_values(["frame", "t"]).groupby("n") if trajectory_is_valid(t)]
            indices = np.array([_[0] for _ in trajectories_indices])
            trajectories = [_[1] for _ in trajectories_indices]

            predictions['n'] = indices
            print(predictions)

            #predictions.to_csv('gratin_results_for_n2_+myoIb_1_trackedPar.csv')
            #predictions.to_csv('gratin_results_Video6_test.csv')

            pred_name = 'gratin_results_for_'+file_name.split('/')[-1].split('.')[0]+'.csv'
            predictions.to_csv(pred_name)

    else:
        trajectories = pd.read_csv(filename)
        model, encoder = load_model(export_path="tracktor2D")

        predictions = get_predictions(model, encoder, trajectories)

        if plot:
            plot_demo(
            model,
            encoder,
            length_range = (7, 55), # these values can differ from those used during training
            noise_range = (0.015, 0.05)
            )

        trajectories_indices = [(n, t[["x", "y"]].values, t["t"].values) for n, t in trajectories.sort_values(["frame", "t"]).groupby("n") if trajectory_is_valid(t)]
        indices = np.array([_[0] for _ in trajectories_indices])
        trajectories = [_[1] for _ in trajectories_indices]

        predictions['n'] = indices
        print(predictions)

        #predictions.to_csv('gratin_results_for_n2_+myoIb_1_trackedPar.csv')
        #predictions.to_csv('gratin_results_Video6_test.csv')

        pred_name = 'gratin_results_for_'+filename.split('/')[-1].split('.')[0]+'.csv'
        predictions.to_csv(pred_name)


file_name=input("Please enter the name of a trajectories file : ")
plot=input("do you want to plot a 2D projection of the latent space representation of your input data (y/n)? ")
if plot=='y':
    easy_predictions(file_name,True)
else:
    easy_predictions(file_name)
