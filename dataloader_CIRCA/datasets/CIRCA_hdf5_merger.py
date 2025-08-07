import os
import sys

import h5py
from tqdm import tqdm


def merge_hdf5_files(source_dir, output_file):
    """
    Fusionne les fichiers HDF5 en regroupant les données par clé MGRS.

    Ce script parcourt les fichiers HDF5 d'un répertoire source. Pour chaque
    fichier, il lit les groupes de premier niveau (clés MGRS) et copie les
    sous-groupes (clés MGRSC) dans le groupe MGRS correspondant du fichier
    de sortie. Si un groupe MGRS n'existe pas dans le fichier de sortie,
    il est créé.

    Args:
        source_dir (str): Le chemin vers le répertoire contenant les fichiers HDF5.
        output_file (str): Le chemin vers le fichier HDF5 de sortie à créer.
    """
    # Vérifier si le répertoire source existe
    if not os.path.isdir(source_dir):
        print(f"Erreur : Le répertoire source n'existe pas : {source_dir}")
        return

    # Lister tous les fichiers .hdf5 dans le répertoire source
    try:
        hdf5_files = sorted([f for f in os.listdir(source_dir) if f.endswith((".hdf5", ".h5"))])
    except FileNotFoundError:
        print(f"Erreur : Impossible d'accéder au répertoire : {source_dir}")
        return

    if not hdf5_files:
        print(f"Aucun fichier .hdf5 trouvé dans {source_dir}")
        return

    print(f"{len(hdf5_files)} fichiers HDF5 trouvés. Début de la fusion dans {output_file}...")

    # Créer le répertoire de sortie s'il n'existe pas
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Ouvrir le fichier de sortie en mode écriture
    with h5py.File(output_file, "w") as h5_out:
        # Utiliser tqdm pour une barre de progression
        for filename in tqdm(hdf5_files, desc="Fusion des fichiers"):
            file_path = os.path.join(source_dir, filename)
            try:
                with h5py.File(file_path, "r") as h5_in:
                    # Parcourir les clés de premier niveau (MGRS_id) du fichier source
                    for mgrs_id in h5_in.keys():
                        # Obtenir ou créer le groupe MGRS de destination dans le fichier de sortie
                        dest_mgrs_group = h5_out.require_group(mgrs_id)
                        source_mgrs_group = h5_in[mgrs_id]

                        # Parcourir les clés de deuxième niveau (MGRSC_id)
                        for mgrsc_id in source_mgrs_group.keys():
                            # Vérifier si la zone MGRSC existe déjà pour éviter les écrasements
                            if mgrsc_id in dest_mgrs_group:
                                print(
                                    f"Avertissement : La zone MGRSC '{mgrsc_id}' existe déjà "
                                    f"dans le groupe MGRS '{mgrs_id}'. Ignoré. (Fichier: {filename})"
                                )
                                continue

                            # Copier le groupe MGRSC du fichier source vers le groupe MGRS de destination
                            source_mgrs_group.copy(mgrsc_id, dest_mgrs_group)
            except Exception as e:
                print(f"\nErreur lors du traitement du fichier {filename}: {e}")
                # Optionnel : supprimer le fichier de sortie en cas d'erreur
                # os.remove(output_file)
                # sys.exit(1)

    print(f"\nFusion terminée. Le fichier de sortie est : {output_file}")


if __name__ == "__main__":
    # --- Paramètres à configurer ---
    # Répertoire contenant les fichiers HDF5 à fusionner
    # SOURCE_DIRECTORY = "/DATA_10TB/data_rpg/circa/hdf5/archives_MGRSC"
    SOURCE_DIRECTORY = "/DATA_10TB/data_rpg/circa/hdf5/test_merge"
    # Chemin complet du fichier HDF5 fusionné en sortie
    OUTPUT_HDF5_FILE = "/DATA_10TB/data_rpg/circa/hdf5/merged_archives.hdf5"

    # Lancer la fonction de fusion
    merge_hdf5_files(SOURCE_DIRECTORY, OUTPUT_HDF5_FILE)
