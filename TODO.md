# TODO LIST

---

### 🎯 Data
- [ ] **Créer le fichier hdf5 contenant le dataset avec toutes les données S2 / S1 ASC / DESC**
    - [ ] Lancement du script `CIRCA_hdf5_maker.py` et attente de voir si le script run bien
    - [ ] Observer les fichiers obtenues et voir s'ils sont bien exploitables.

---

### 📚 Datasets
- [ ] **Adapter les codes des datasets pour qu'ils puissent bien prendre en compte les nouvelles données ASC/DESC**
    - [ ] Adaptation du fichier CIRCA_hdf5_reader.py
    - [ ] Adaptation du fichier UTILISE_adapter.py
    - [ ] Faire que le cas `closest mix date ASC/DESC` fonctionne car (dim=4). Lancer un training pour voir si cela fonctionne correctement.
    - [ ] Faire que le cas `closest` avec juste des données ASC ou DESC fonctionne car dim=4 dans un cas où dans l'autre. Penser à bien regarder les dict d'appariement pour voir s'il n'y a pas d'erreur.
    - [ ] Faire que le cas d'utilisation `full S1 ASC + DESC` fonctionne ici dim=4. Lancer un training avec ce nouveau cas.
    - [ ] Faire une comparaison avec les trainings fait auparavant (cf nécessité du code de métriques)
    - [ ] Nettoyage des codes ne servant plus (pousser avec les backups olds puis delete)

---

### 🚚 Dataloaders
- [ ] **Vérifications de la variétés des dates sélectionnées d'une epoch à une autre.** Regarder potentiellement aussi si les images paraissent bonnes et permettent d'être observé.

---

### 📊 Metrics
- [ ] **Avancer sur le code pour les inférences: aussi bien métriques que visuels**
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
    - [ ] `MS full`
    - [ ] `MS full + SAR mix`
    - [ ] `MS full + SAR ASC + DESC`
    - [ ] ==> détermination de la meilleur utilisation des données S1 ? 
    - [ ] `RGB - NIR` (10m res only)
    - [ ] `RGB - NIR` (10m res only) avec la meilleur méthode d'utilisation des données SAR.
    - [ ] ==> choix entre `MS full` où `RGB-NIR` 
    - [ ] Training `MS full / RGB-NIR` avec des dates complètements masquées pour voir si le modèle arrive à reconstruire sans trop prendre en compte les géométries des masques des nuages synthétiques.
    - [ ] Finetuning des hyperparamètres afin d'améliorer le modèles

---

### 🛠️ MISC: amélioration du pipeline de training / modèle
- [ ] La gestion des sequences de longueurs différentes est-elle bien faite ?
- [ ] refaire une passe sur la génération de masks dans le getitem du dataloader.



