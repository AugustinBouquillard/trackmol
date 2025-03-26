import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as colors




# Charger les fichiers
def axon_or_synapse_data(locs_path = "data/experimental_data/0deb5656cadde25b.locs", results_path = "data/experimental_data/gratin_results_for_0deb5656cadde25b.csv"):

    # Lire les fichiers CSV
    gratin_results = pd.read_csv(results_path)
    locs_data = pd.read_csv(locs_path)

    # Conserver uniquement les colonnes utiles de locs_data
    locs_subset = locs_data[['n', 'on_axon', 'on_synapse']]

    # Supprimer les doublons pour éviter les erreurs lors de la jointure
    locs_subset = locs_subset.drop_duplicates(subset='n')

    # Effectuer la jointure sur la colonne 'n'
    merged_data = gratin_results.merge(locs_subset, on='n', how='left')

    # Renommer les colonnes pour plus de clarté
    merged_data.rename(columns={'on_axon': 'is_axon', 'on_synapse': 'is_synapse'}, inplace=True)

    # Sauvegarder le fichier CSV résultant
    output_path = "data/experimental_data/updated_gratin_results.csv"
    merged_data.to_csv(output_path, index=False)

    print(f"Le fichier mis à jour a été sauvegardé sous '{output_path}'")
    
    
    
