# TODO LIST
---
Débugger le problème avec le code sur les métriques
Faire l'adaptation des métriques avec les masques de Célestin
Unifier le problème de fuite mémoire avec le détach seulement dans le cr_torchmetrics.py 
Comparer rapidement les sorties entre r2_score_pytorch et sklearn.r2

### 🎯 Data
- [ ] **Créer le fichier hdf5 contenant le dataset avec toutes les données S2 / S1 ASC / DESC**
    - [X] Comprendre pourquoi le script n'a pas fonctionné jusqu'au bout => problème avec le STORE-DAI ... / problème d'accumulatoin de RAM ? 
    - [X] Tester le script de merge des fichiers hdf5 pour voir si tout fonctionne correctement.
    - [X] Observer les fichiers obtenues et voir s'ils sont bien exploitables.
    - [X] Refaire la zone MGRS 31TDJ
    - [X] Copie en entre la machine LNV87 et le store-dai 
    - [X] Copie entre le store et les noeuds jzellou..
    - [X] Transférer les données vers jzay
    
---

### 📚 Datasets
- [ ] **Adapter les codes des datasets pour qu'ils puissent bien prendre en compte les nouvelles données ASC/DESC**
    - [X] Adaptation du fichier CIRCA_hdf5_reader.py (en partie faite)
    - [X] Adaptation du fichier UTILISE_adapter.py (en partie faite)
    - [X] Vérifications des num_classes !!!URGENT!!! (dans le dataset OK !!!)
    ###### Test des trainings arrivent à tourner avec ces dataloaders
    - [X] Création de nouveaux fichiers de configs
    - [X] Faire que le cas `mix_closest date ASC/DESC` fonctionne car (dim=4). Lancer un training pour voir si cela fonctionne correctement.
    - [X] Faire que le cas `mix_closest` avec juste des données ASC ou DESC fonctionne car dim=4 dans un cas où dans l'autre. Penser à bien regarder les dict d'appariement pour voir s'il n'y a pas d'erreur.
    - [X] Faire que le cas d'utilisation `full S1 ASC + DESC` fonctionne ici dim=8. Lancer un training avec ce nouveau cas.
    - [X] Nettoyage des codes ne servant plus (pousser avec les backups olds puis delete)
---

### 🚚 Dataloaders
- [ ] **Vérifications de la variétés des dates sélectionnées d'une epoch à une autre.** 
!!!URGENT!!!
Regarder potentiellement aussi si les images paraissent bonnes et permettent d'être observé.
- [X] Vérification des t_sampled pour la même TS, (correspondance avec les dates marquées comme valides)
- [X] Regarder si le paramètre render_occluded_above_p change qch dans la création de masks 
- [ ] Refaire une passe sur la génération de masks dans le getitem du dataloader. (est-ce que la génération de mask n'est-elle pas trop faible..)

---

### 📊 Metrics
- [ ] **Avancer sur le code pour les inférences: aussi bien métriques que visuels**
    - [ ] Dev code de test sur les zones MGRS test => prise en compte full data et faire en sorte de sortir des métriques occluded / observed !!!URGENT!!!

    - [ ] Côté visuel : regarder la gestion des données de Célestin avec le + 150. 
    - [ ] Reconstruction `full MS` avec une observation / une TS complète (metrics + objectif de visu)
    - [ ] Reconstruction `full MS` obtention des métriques pour un test complets.
    - [ ] Reconstruction seulement des bandes à `RGB-NIR` (10m resolution seulement.) 
    - [ ] Pour chacun des vérifier que les métriques soient faites selon les manières suivantes:
        - [ ] métriques spécifiques aux dates avec nuages synthétiques qui seront reconstruites
        - [ ] métriques spécifiques aux dates sans nuages syn pour voir à quel point le modèle ne les altèrent pas.
        - [ ] métriques OA pour avoir une représentation des perfs du modèle sur l'ensemble des dates.
        - [ ] appliquer les mêmes métriques avec les masks de Célestin et les dates complètement cachées.
        - [ ] same avec seulement les pixels appartenant à des parcelles agricoles présentes dans le RPG.
    - [ ] Faire un code permettant de reconstruire le NDVI à partir des données S2 et selon certaines dates S1 (coïncidant avec les dates S1 utilisées par Célestin) afin de pouvoir comparer nos résultats.

---

### 🚀 Trainings
- [ ] **Lancer des trainings avec les différents cas à expérimenter:**
    - [X] Implémentation de la méthode avec masquage complet d'une date de la TS 
    - [ ] `MS full`
    - [ ] `MS full + SAR mix`
    - [ ] `MS full + SAR ASC + DESC`
    - [ ] ==> détermination de la meilleur utilisation des données S1 ? 
    - [ ] `RGB - NIR` (10m res only)
    - [ ] `RGB - NIR` (10m res only) avec la meilleur méthode d'utilisation des données SAR.
    - [ ] ==> choix entre `MS full` où `RGB-NIR` 
    - [ ] Training `MS full / RGB-NIR` avec des dates complètements masquées pour voir si le modèle arrive à reconstruire sans trop prendre en compte les géométries des masques des nuages synthétiques.
    - [ ] Finetuning des hyperparamètres afin d'améliorer le modèles
    - [ ] Faire une comparaison avec les trainings fait auparavant (cf nécessité du code de métriques)

---

### 🛠️ MISC: amélioration du pipeline de training / modèle
- [X] La gestion des sequences de longueurs différentes est-elle bien faite ? !!!URGENT!!! Non pour les min_seq_lengths ... (tri fait lors de la création du hdf5 avec une longueur min de 10.)
- [X] Regarder si la gestion des min_seq_length et max_seq_length est bien faite
- [X] Est-ce que la feature permettant de faire l'utilisation des dates consécutives est vraiment intéressante dans notre cas. Si oui, la remettre en place dans le code du Dataset. => None Pas le temps

### Annexe si temps
- [ ] Mettre l'affichage de config avec rich comme dans le repo UnCRtainTS
- [X] Comprendre dans le pairing de dates S2/S1 les index ne sont pas bons? => mauvaise idée de prendre l'index de la nouvelle liste, il aurait fallu pas refaire l'index mais mettre direct la valeur.
- [ ] Potentiellement refaire les fichiers hdf5 avec les TS de moins de 10 dates? 
- [ ] Explorer ce que fait le paramètre -mask dans la sélection des channels.