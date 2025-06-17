# Explorateur HDF5 Générique

## Description

Script Python générique pour explorer l'architecture de n'importe quel fichier HDF5 avec une interface riche et colorée utilisant Rich.

## Fonctionnalités

- ✅ **Statistiques globales** : Nombre de datasets, taille totale, types de données
- ✅ **Arborescence visuelle** : Structure hiérarchique interactive avec limitation intelligente
- ✅ **Analyse détaillée** : Focus sur un groupe de données représentatif
- ✅ **Interface moderne** : Affichage coloré avec tableaux, arbres et barres de progression
- ✅ **Options modulaires** : Contrôle précis des sections à afficher
- ✅ **Gestion d'arguments** : Ligne de commande ou détection automatique

## Installation

```bash
pip install h5py rich
```

## Utilisation

### Ligne de commande

```bash
# Analyser un fichier spécifique
python explore_hdf5_complete.py mon_fichier.hdf5

# Recherche automatique (cherche *.hdf5 dans le répertoire courant)
python explore_hdf5_complete.py
```

### Utilisation programmatique

```python
from explore_hdf5_complete import explore_hdf5_complete

# Exploration complète
explore_hdf5_complete("data.hdf5")

# Statistiques seulement
explore_hdf5_complete("data.hdf5", show_tree=False, show_stats=True, show_data_group=False)

# Structure + détails de groupe
explore_hdf5_complete("data.hdf5", show_tree=True, show_stats=False, show_data_group=True)
```

## Paramètres

- `show_tree` : Affiche l'arborescence hiérarchique (défaut: True)
- `show_stats` : Affiche les statistiques globales (défaut: True)  
- `show_data_group` : Affiche les détails d'un groupe représentatif (défaut: True)

## Icônes utilisées

- 📊 Datasets multidimensionnels (3D+)
- 🗂️ Datasets 2D (matrices)
- 📈 Datasets 1D (vecteurs, séries temporelles)
- 📁 Groupes HDF5
- 📄 Datasets scalaires

## Exemples de sortie

Le script affiche :
1. **Informations générales** : Taille du fichier, clés racines
2. **Statistiques** : Nombre de datasets, types de données, plus gros éléments
3. **Arborescence** : Structure hiérarchique avec limitation intelligente
4. **Détails** : Analyse d'un groupe représentatif avec ses datasets

## Compatibilité

- ✅ Fonctionne avec n'importe quel fichier HDF5
- ✅ Agnostique au domaine (scientifique, ML, géospatial, etc.)
- ✅ Gestion robuste des gros fichiers avec limitation d'affichage
- ✅ Support Python 3.7+

## Limites configurables

- `MAX_DISPLAYED_ITEMS = 10` : Nombre max d'items affichés par groupe
- `MAX_SEARCH_DEPTH = 3` : Profondeur de recherche des groupes
- `MAX_SEARCH_ITEMS = 3` : Items examinés par niveau

## Gestion d'erreurs

- Vérification de l'existence du fichier
- Gestion des permissions insuffisantes  
- Messages d'erreur clairs et colorés
- Aide automatique si aucun fichier spécifié
