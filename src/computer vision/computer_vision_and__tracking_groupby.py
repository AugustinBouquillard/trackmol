# -*- coding: utf-8 -*-
"""
Fichier qui réalise le traitement des vidéos des observations et qui réalise le tracking
"""
import cv2
import numpy as np
import tifffile as tiff
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import trackpy as tp
import plotly.graph_objects as go
from tqdm import tqdm  # Import de tqdm pour la barre de progression
import random

def detect_circles_in_tiff(file_path, output_path, max_frames_to_display=10, min_radius=5, max_radius=30):
    """
    Détecte des cercles dans un fichier .tif et sauvegarde les masques détectés.

    Paramètres :
        file_path (str) : Chemin vers le fichier .tif d'entrée.
        output_path (str) : Chemin pour sauvegarder le fichier .tif contenant les masques de cercles détectés.
        max_frames_to_display (int) : Nombre maximum de frames à afficher pour visualisation.
        min_radius (int) : Rayon minimal des cercles à détecter (en pixels).
        max_radius (int) : Rayon maximal des cercles à détecter (en pixels).
    """
    # Charger toutes les frames du fichier .tif
    tiff_stack = tiff.imread(file_path)

    # Vérifier si plusieurs frames sont présentes
    if len(tiff_stack.shape) != 3:
        print("Le fichier .tif ne contient pas de frames multiples.")
        return

    # Créer une liste pour stocker les masques de cercles détectés
    circle_masks = []

    # Créer une barre de progression pour suivre l'avancement dans le terminal
    with tqdm(total=len(tiff_stack), desc="Traitement des frames par computer vision", unit="frame") as pbar:
        # Parcourir chaque frame
        for frame_idx, frame in enumerate(tiff_stack):
            # Normaliser la frame pour l'affichage et le traitement
            normalized_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

            # Convertir en niveaux de gris (si ce n'est pas déjà le cas)
            gray = normalized_frame

            # Appliquer un seuil pour segmenter l'image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Appliquer un flou gaussien pour réduire le bruit
            blurred = cv2.GaussianBlur(binary, (9, 9), 2)

            # Détection des cercles avec Hough Circle Transform
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                       param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)

            # Créer un masque pour les cercles détectés
            circle_mask = np.zeros_like(binary)
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                if frame_idx == 0:
                    r0 = 10
                for (x, y, r) in circles:
                    cv2.circle(circle_mask, (x, y), r0//2, 255, thickness=-1)  # Cercle rempli

            # Ajouter le masque de cercles à la liste
            circle_masks.append(circle_mask)

            # Afficher les frames et leurs masques détectés (si dans la limite spécifiée)
            # if frame_idx < max_frames_to_display:
            #     plt.figure(figsize=(12, 6))
            #     plt.subplot(1, 2, 1)
            #     plt.title(f"Frame Originale {frame_idx+1}")
            #     plt.imshow(gray, cmap="gray")
            #     plt.axis("off")

            #     plt.subplot(1, 2, 2)
            #     plt.title(f"Masque de Cercles {frame_idx+1}")
            #     plt.imshow(circle_mask, cmap="gray")
            #     plt.axis("off")
            #     plt.show()

            # Mise à jour de la barre de progression
            pbar.update(1)

    # Convertir les masques en tableau numpy
    circle_masks_stack = np.array(circle_masks)

    # Sauvegarder le stack des masques en tant que fichier .tif
    tiff.imwrite(output_path, circle_masks_stack)
    print(f"Les masques des cercles détectés ont été sauvegardés sous : {output_path} \n")



def tif_to_mp4(tif_input, output_video, fps=30):
    """
    Convertit un fichier .tif multi-frame en une vidéo MP4 avec une barre d'avancement.
    
    Paramètres:
    ----------
    tif_input : str
        Chemin du fichier .tif contenant les frames.
    output_video : str
        Chemin de sortie pour la vidéo MP4.
    fps : int, optionnel
        Framerate (images par seconde) de la vidéo. Par défaut, 30.
    """
    # Charger les frames du fichier .tif
    frames = tiff.imread(tif_input)
    print(f"Nombre de frames détectées : {len(frames)}")
    
    # Obtenir les dimensions de la première frame
    height, width = frames[0].shape
    
    # Initialiser le writer vidéo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec pour MP4
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Ajouter chaque frame à la vidéo avec une barre de progression
    for i, frame in tqdm(enumerate(frames), total=len(frames), desc="Création de la vidéo", ncols=100):
        # Normaliser les valeurs de la frame (si nécessaire)
        frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame_normalized = frame_normalized.astype('uint8')
        
        # Convertir en RGB si les frames sont en niveaux de gris
        if len(frame_normalized.shape) == 2:  # Si l'image est en niveaux de gris
            frame_normalized = cv2.cvtColor(frame_normalized, cv2.COLOR_GRAY2BGR)
        
        # Ajouter la frame à la vidéo
        video_writer.write(frame_normalized)
    
    # Libérer les ressources
    video_writer.release()
    print(f"Vidéo MP4 générée : {output_video} \n")


def combine_videos_side_by_side(video1_path, video2_path, output_path, fps=30):
    """
    Combine deux vidéos côte à côte et les enregistre dans un fichier MP4, avec une barre d'avancement.
    
    Paramètres:
    ----------
    video1_path : str
        Chemin de la première vidéo.
    video2_path : str
        Chemin de la deuxième vidéo.
    output_path : str
        Chemin de sortie pour la vidéo combinée.
    fps : int, optionnel
        Framerate (images par seconde) de la vidéo combinée. Par défaut, 30.
    """
    # Charger les vidéos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    # Vérifier que les vidéos sont ouvertes
    if not cap1.isOpened() or not cap2.isOpened():
        raise ValueError("Impossible de lire l'une ou l'autre des vidéos.")
    
    # Obtenir les dimensions des vidéos
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Redimensionner la deuxième vidéo pour qu'elle ait la même hauteur que la première
    if height1 != height2:
        scale = height1 / height2
        width2 = int(width2 * scale)
        height2 = height1

    # Dimensions de la vidéo combinée
    combined_width = width1 + width2
    combined_height = height1
    
    # Obtenir le nombre total de frames des vidéos
    total_frames = int(min(cap1.get(cv2.CAP_PROP_FRAME_COUNT), cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    # Initialiser le writer vidéo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec pour MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))
    
    # Traiter les frames des vidéos avec une barre d'avancement
    with tqdm(total=total_frames, desc="Combinaison des vidéos", ncols=100) as pbar:
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            # Si une des vidéos est terminée, arrêter la lecture
            if not ret1 or not ret2:
                break
            
            # Redimensionner la deuxième frame
            frame2_resized = cv2.resize(frame2, (width2, height2))
            
            # Combiner les frames côte à côte
            combined_frame = np.hstack((frame1, frame2_resized))
            
            # Ajouter la frame combinée au writer
            out.write(combined_frame)
            
            # Mettre à jour la barre d'avancement
            pbar.update(1)
    
    # Libérer les ressources
    cap1.release()
    cap2.release()
    out.release()
    print(f"Vidéo combinée générée : {output_path} \n")


def estimate_diameter_from_tiff(tiff_file, threshold=127, debug=False):
    """
    Estime automatiquement le diamètre optimal des billes pour chaque frame d'un fichier OME-TIFF.
    
    Paramètres :
    ------------
    tiff_file : str
        Chemin vers le fichier .ome.tif contenant les frames.
    threshold : int
        Le seuil pour binariser l'image (0-255).
    debug : bool
        Si True, affiche les étapes de détection pour la première frame et les statistiques globales.

    Retourne :
    ----------
    list : Liste des diamètres moyens estimés pour chaque frame.
    """
    # Charger le fichier TIF multi-frames
    try:
        stack = tiff.imread(tiff_file)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier TIF : {e}")
        return None

    # Liste pour stocker les diamètres moyens de chaque frame
    avg_diameters = []

    # Traiter chaque frame du fichier
    for frame_index, image in enumerate(stack):
        # Convertir l'image en 8 bits si nécessaire
        if image.dtype != np.uint8:
            image = cv2.convertScaleAbs(image, alpha=(255.0 / image.max()))  # Normaliser en 8 bits
        
        # Binarisation de l'image
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        
        # Détection des contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calcul des diamètres
        diameters = []
        for contour in contours:
            # Obtenir un cercle englobant chaque bille
            (_, _), radius = cv2.minEnclosingCircle(contour)
            diameter = 2 * radius
            diameters.append(diameter)
        
        # Calcul du diamètre moyen pour cette frame
        avg_diameter = int(np.mean(diameters)) if diameters else 0
        avg_diameters.append(avg_diameter)
        
        if debug and frame_index == 0:  # Afficher uniquement pour la première frame
            print(f"Diamètres détectés (frame {frame_index}): {diameters}")
            print(f"Diamètre moyen estimé (frame {frame_index}): {avg_diameter} pixels")
            # Afficher l'image binaire avec les contours
            plt.imshow(binary, cmap='gray')
            plt.title(f'Contours détectés - Frame {frame_index}')
            plt.show()

    if debug:
        print(f"Diamètres moyens pour toutes les frames : {avg_diameters}")

    return avg_diameters

#estimate_diameter_from_tiff("film6_3dot2um_TRITC_40X_10Hz_MMStack.ome.tif", threshold=127, debug=True)


def track(tif_input, output, diameter = 5, minmass = 50, search_range = 50, memory = 2):
    """
    Cette fonction détecte et suit des caractéristiques (particules) dans une séquence d'images TIFF et 
    sauvegarde les trajectoires détectées dans un fichier CSV.

    Paramètres:
    ----------
    tif_input (str) : Chemin du fichier TIFF contenant les images à traiter. Ce fichier doit être un fichier
                      multi-frame (une pile d'images).
    output (str) : Chemin du fichier de sortie où les résultats des trajectoires seront sauvegardés (au format CSV).

    Le processus de la fonction se déroule en plusieurs étapes :
    1. Charger les images du fichier TIFF.
    2. Détecter les caractéristiques dans chaque frame en utilisant l'algorithme de détection de Trackpy.
    3. Lier les caractéristiques entre les frames pour créer des trajectoires.
    4. Visualiser les trajectoires sur l'ensemble de la séquence d'images.
    5. Sauvegarder les trajectoires dans un fichier CSV pour une analyse ultérieure.
    """

    # Charger les images TIFF
    tiff_stack = tiff.imread(tif_input)
    
    # Détection des caractéristiques
    features = []
    print("Détection des caractéristiques...")
    for frame_idx, frame in tqdm(enumerate(tiff_stack), total=len(tiff_stack), desc="Traitement des frames pour le tracking"):
        # Localiser les caractéristiques
        f = tp.locate(frame, diameter, minmass=minmass)
        f['frame'] = frame_idx  # Ajouter l'indice de frame
        features.append(f)
    
    # Combiner les détections dans un DataFrame
    all_features = pd.concat(features, ignore_index=True)
    
    # Suivi des caractéristiques
    print("Lien des caractéristiques pour générer des trajectoires...")
    trajectories = tp.link_df(all_features, search_range=search_range, memory=memory)

    def merge_trajectories(df, distance_threshold, frame_gap):
        # Trier les trajectoires par particule et frame
        df = df.sort_values(['particle', 'frame']).reset_index(drop=True)
        merged_particles = {}
    
        # Parcourir les trajectoires existantes
        for particle_id in df['particle'].unique():
            traj = df[df['particle'] == particle_id]
            last_row = traj.iloc[-1]
            candidates = df[(df['frame'] > last_row['frame']) & 
                            (df['frame'] <= last_row['frame'] + frame_gap)]
            
            # Vérifier la distance avec les trajectoires candidates
            for _, row in candidates.iterrows():
                dist = ((last_row['x'] - row['x'])**2 + (last_row['y'] - row['y'])**2)**0.5
                if dist < distance_threshold:
                    merged_particles[row['particle']] = particle_id
    
        # Mettre à jour les particules fusionnées
        df['particle'] = df['particle'].replace(merged_particles)
        return df

    trajectories = merge_trajectories(trajectories, distance_threshold=10, frame_gap=5)

     # Filtrer les trajectoires trop courtes (par exemple, <10 frames)
    print("Filtrage des trajectoires courtes...")
    trajectories['particle'] = trajectories['particle'].astype('category')  # Faciliter le filtrage
    long_trajectories = trajectories.groupby('particle').filter(lambda x: len(x) >= 10)
    
    # Sauvegarder les trajectoires filtrées dans un fichier CSV
    long_trajectories.to_csv(output, index=False)
    print(f"Les trajectoires filtrées ont été sauvegardées dans '{output}'")
    
    
def filter_trajectories_by_frames(input_csv, output_csv, min_frames=150):
    """
    Filtre les trajectoires des particules pour conserver uniquement celles présentes sur au moins `min_frames` frames consécutifs.

    Paramètres :
    ------------
    input_csv : str
        Chemin du fichier CSV d'entrée contenant les trajectoires (avec colonnes 'n' et 'frame').
    output_csv : str
        Chemin du fichier CSV où sauvegarder les trajectoires filtrées.
    min_frames : int
        Nombre minimum de frames consécutifs requis pour conserver une trajectoire.

    Retour :
    --------
    Aucun. Les résultats sont sauvegardés dans `output_csv`.
    """
    # Charger les données
    data = pd.read_csv(input_csv)

    # Vérifier que les colonnes nécessaires existent
    if not {'n', 'frame'}.issubset(data.columns):
        raise ValueError("Le fichier CSV doit contenir les colonnes 'n' et 'frame'.")

    # Regrouper par particule ('n') et calculer le nombre de frames consécutifs
    filtered_particles = []
    for particle_id, group in data.groupby('n'):
        group = group.sort_values(by='frame')  # Trier par frame

        # Vérifier les frames consécutifs
        consecutive_frames = (group['frame'].diff().fillna(1) == 1).cumsum()
        max_consecutive_length = consecutive_frames.value_counts().max()

        if max_consecutive_length >= min_frames:
            filtered_particles.append(group)

    # Combiner les trajectoires filtrées
    filtered_data = pd.concat(filtered_particles, ignore_index=True)

    # Sauvegarder les résultats
    filtered_data.to_csv(output_csv, index=False)
    print(f"Les trajectoires filtrées ont été sauvegardées dans : {output_csv}")


def filter_longest_trajectory(input_csv, output_csv):
    """
    Filtre les trajectoires des particules pour conserver uniquement la trajectoire la plus longue (en termes de frames).

    Paramètres :
    ------------
    input_csv : str
        Chemin du fichier CSV d'entrée contenant les trajectoires (avec colonnes 'n' et 'frame').
    output_csv : str
        Chemin du fichier CSV où sauvegarder la trajectoire la plus longue.

    Retour :
    --------
    Aucun. Les résultats sont sauvegardés dans `output_csv`.
    """
    # Charger les données
    data = pd.read_csv(input_csv)

    # Vérifier que les colonnes nécessaires existent
    if not {'n', 'frame'}.issubset(data.columns):
        raise ValueError("Le fichier CSV doit contenir les colonnes 'n' et 'frame'.")

    # Regrouper par particule ('n') et calculer la longueur de chaque trajectoire
    longest_trajectory = None
    max_length = 0

    for particle_id, group in data.groupby('n'):
        trajectory_length = len(group)
        if trajectory_length > max_length:
            max_length = trajectory_length
            longest_trajectory = group

    # Sauvegarder la trajectoire la plus longue
    if longest_trajectory is not None:
        longest_trajectory.to_csv(output_csv, index=False)
        print(f"La trajectoire la plus longue a été sauvegardée dans : {output_csv}")
    else:
        print("Aucune trajectoire trouvée dans les données.")

def suppr_overall_mvmt(fichier):

    df = pd.read_csv(fichier)
    
    # Trouver la valeur initiale de 'x' pour chaque 'particle' à la première 'frame'
    initial_x = df.groupby('particle').apply(lambda group: group.loc[group['frame'].idxmin(), 'x']).rename('x0')
    initial_y = df.groupby('particle').apply(lambda group: group.loc[group['frame'].idxmin(), 'y']).rename('y0')
    
    # Ajouter la colonne 'x0' au DataFrame
    df = df.merge(initial_x, on='particle')
    df = df.merge(initial_y, on='particle')
    
    # Calculer la différence entre 'x' et 'x0'
    df['dx'] = df['x'] - df['x0']
    df['dy'] = df['y'] - df['y0']
    
    moyennes_deplacements = df.groupby('frame')[['dx', 'dy']].mean().reset_index()
    moyennes_deplacements.rename(columns={'dx': 'mean_dx', 'dy': 'mean_dy'}, inplace=True)
    
    # Joindre les moyennes des déplacements au DataFrame
    df = df.merge(moyennes_deplacements, on='frame')

    # Retirer la moyenne des déplacements
    df['x'] = df['x'] - df['mean_dx']
    df['y'] = df['y'] - df['mean_dy']
    
    df.to_csv('fichier_corrige.csv', index=False)

def csv_for_gratin(fichier, pixel, temps):
  """
    Convertit les coordonnées des particules d'un fichier CSV de pixels à une unité réelle (par exemple, micromètres),
    et ajuste les temps en fonction de l'intervalle entre chaque frame. Sauvegarde les résultats modifiés dans un nouveau fichier CSV.

    Paramètres:
    fichier (str) : Le chemin d'accès au fichier CSV contenant les données de suivi des particules.
                    Ce fichier doit contenir des colonnes "x", "y", "particle" (numéro de la particule) et "frame" (numéro de la frame).
    pixel (float) : Le facteur de conversion pour convertir les coordonnées des particules de pixels à une unité réelle.
                    Par exemple, si chaque pixel représente 0,5 micromètre, mettre pixel = 0.5.
    temps (float) : L'intervalle de temps entre deux frames successives, en secondes. Par exemple, si chaque frame est séparée de 0,1 seconde,
                    mettre temps = 0.1.

    Fonctionnement :
    - La fonction charge les données du fichier CSV.
    - Elle calcule le temps en secondes pour chaque frame (en multipliant le numéro de la frame par l'intervalle de temps `temps`).
    - Elle convertit les coordonnées `x` et `y` de pixels à l'unité réelle (par exemple, micromètres) en les multipliant par le facteur `pixel`.
    - Elle crée une nouvelle colonne `n` pour le numéro de la particule.
    - Elle garde seulement les colonnes pertinentes : `x`, `y`, `n`, `t` (temps), et `frame`.
    - Enfin, elle sauvegarde les résultats dans un nouveau fichier CSV.
    """
    
  df = pd.read_csv(fichier)
  df["t"] = df["frame"] * temps
  df["x"] = df["x"] * pixel
  df["y"] = df["y"] * pixel
  df["n"] = df["particle"]
  entetes = ["x", "y", "n", "t", "frame"]

  df = df[entetes]
  df.to_csv("tracktor_film5_"+fichier, index=False)
  


def plot_particle_trajectories(input_csv_path):
    """
    Affiche les trajectoires des particules à partir d'un fichier CSV contenant les résultats du suivi.

    Cette fonction charge un fichier CSV contenant les coordonnées des particules et leur identifiant
    pour créer un graphique interactif avec Plotly. Chaque particule sera représentée par une ligne
    indiquant son trajet en fonction des coordonnées X et Y.

    Paramètres :
    -----------
    input_csv_path : str
        Le chemin d'accès au fichier CSV contenant les données des trajectoires des particules.
        Le fichier CSV doit contenir les colonnes suivantes :
        - 'particle' : identifiant unique de chaque particule
        - 'x' : coordonnée X des particules à chaque instant
        - 'y' : coordonnée Y des particules à chaque instant

    Exemple :
    --------
    plot_particle_trajectories("tracking_results.csv")
    """
    
    # Charger le fichier CSV contenant les résultats du suivi
    trajectories = pd.read_csv(input_csv_path)

    # Créer un graphique Plotly pour afficher les trajectoires
    fig = go.Figure()

    # Obtenir la liste des particules uniques
    unique_particles = trajectories['particle'].unique()

    # Ajouter les trajectoires des particules avec une barre de progression
    for particle_id in tqdm(unique_particles, desc="Ajout des trajectoires au graphique intéractif"):
        # Extraire les trajectoires pour la particule spécifique
        particle_trajectory = trajectories[trajectories['particle'] == particle_id]

        # Ajouter une trace pour chaque particule
        fig.add_trace(go.Scatter(x=particle_trajectory['x'], y=particle_trajectory['y'],
                                 mode='lines', name=f'Particule {particle_id}'))

    # Ajouter un titre et des labels
    fig.update_layout(
        title="Trajectoires des particules",
        xaxis_title="X",
        yaxis_title="Y",
        showlegend=True
    )

    # Utiliser le rendu dans le navigateur pour garantir la visibilité du graphique
    fig.write_html("trajectoires_particules.html")  # Sauvegarder l'HTML dans un fichier
    print("Le graphique a été enregistré sous le nom 'trajectoires_particules.html'.")
    print("Ouvrez ce fichier dans votre navigateur pour voir le graphique. \n")
    
    # Afficher l'interface interactive de Plotly dans un navigateur
    fig.show()
    
def create_video_with_trajectories(tiff_path, tracking_csv, output_video_path, fps=30):
    """
    Génère une vidéo montrant les trajectoires des particules se dessinant sur les frames originales.

    Paramètres :
    ----------
    tiff_path : str
        Chemin vers le fichier TIFF contenant les frames originales.
    tracking_csv : str
        Chemin vers le fichier CSV contenant les résultats du suivi des particules.
    output_video_path : str
        Chemin de sortie pour la vidéo générée.
    fps : int, optionnel
        Framerate (images par seconde) de la vidéo. Par défaut, 30.
    """
    # Charger les frames du fichier TIFF
    frames = tiff.imread(tiff_path)

    # Charger les données de trajectoires
    trajectories = pd.read_csv(tracking_csv)

    # Obtenir les dimensions des frames
    height, width = frames[0].shape

    # Initialiser le writer vidéo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec pour MP4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Initialiser un dictionnaire pour stocker les positions accumulées des particules
    particle_paths = {}

    # Générer une couleur aléatoire unique pour chaque particule
    particle_colors = {}

    # Ajouter chaque frame à la vidéo avec les trajectoires en construction
    for frame_idx, frame in tqdm(enumerate(frames), total=len(frames), desc="Création de la vidéo avec trajectoires"):
        # Normaliser la frame pour l'affichage
        frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        frame_colored = cv2.cvtColor(frame_normalized, cv2.COLOR_GRAY2BGR)

        # Filtrer les trajectoires pour cette frame
        current_frame_data = trajectories[trajectories['frame'] == frame_idx]

        # Ajouter les nouvelles positions au dictionnaire des trajectoires
        for _, row in current_frame_data.iterrows():
            particle_id = row['particle']
            x, y = int(row['x']), int(row['y'])

            if particle_id not in particle_paths:
                particle_paths[particle_id] = []
                # Attribuer une couleur unique aléatoire à la particule
                particle_colors[particle_id] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )

            particle_paths[particle_id].append((x, y))

        # Dessiner les trajectoires accumulées
        for particle_id, path in particle_paths.items():
            color = particle_colors[particle_id]
            for i in range(1, len(path)):
                cv2.line(frame_colored, path[i-1], path[i], color, thickness=2)

        # Ajouter la frame modifiée au writer vidéo
        video_writer.write(frame_colored)

    # Libérer les ressources
    video_writer.release()
    print(f"Vidéo générée avec trajectoires : {output_video_path}")
    

    

# Test sur le film 5
#tif_to_mp4('random_walk.tif', 'output_test_before_computer_vision.mp4', fps=30)
#detect_circles_in_tiff('film6_3dot2um_TRITC_40X_10Hz_MMStack.ome.tif', 'test.tif', max_frames_to_display=10, min_radius=5, max_radius=30)
#tif_to_mp4('test.tif', 'output_test_post_computer_vision.mp4', fps=30)
#track("test.tif", 'test6.csv', diameter = 65, minmass = 110, search_range =5)
plot_particle_trajectories('test6.csv')
#suppr_overall_mvmt('test6.csv')
#plot_particle_trajectories('fichier_corrige.csv')
#csv_for_gratin('test.csv', 0.5, 0.1)
#combine_videos_side_by_side('output_test_before_computer_vision.mp4', 'output_test_post_computer_vision.mp4', "output_video_side_by_side.mp4", fps=30)
#create_video_with_trajectories(tiff_path='film6_3dot2um_TRITC_40X_10Hz_MMStack.ome.tif',tracking_csv='fichier_corrige.csv',output_video_path='output_with_trajectories.mp4',fps=30)

