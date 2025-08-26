# TODO LIST
- [X] Revoir rapidement la gestion des outputs de sorties
- [X] Test code train pour voir si les tests sont bien réalisés à la fin du training
- [ ] Finir de faire bien marcher le train avec le mode fully masked random
- [ ] Finir de faire bien marcher le train avec le mode fully masked aléatoire
- [ ] Vérifier visuellement ce que renvoie le dataset dans les différents cas consecutive/aleatoire
- [ ] Finir de faire fonctionner le mode fully_masked
- [ ] Adapter le code pour faire varier la génération de mask d'un split à l'autre (cas du train avec une différence entre la validation et le training): par exemple random cloud pour le train et fully masked pour la validation.

---

### Aujourd'hui:
- [X] Mettre en place un système de chunk ? => utilisation du temporal window à l'intérieur du code Imputation.
- [X] Comment se fait le découpage de la TS => comprehension du trimming avec max_seq_length
- [X] Finir l'adaptation du chargement de données aux tests sets
- [ ] Sortir des métriques avec les modèles déjà entrainés sur les nouveaux jeu de tests.
- [ ] Sortir d'avantage d'images d'inférence pour chacun des modèles 
- [ ] Mettre l'étape d'évaluation complète à la fin d'un training (changement dans les seeds)
- [ ] Lancer de nouveaux trainings sur jzellou (=> check si les data ont bien été copié sur les noeuds)
- [ ] *si temps mettre la possiblité de faire une évaluation complète lors des entrainements

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
    - [X] Dev code de test sur les zones MGRS test => prise en compte full data et faire en sorte de sortir des métriques occluded / observed !!!URGENT!!!
    - [X] Côté visuel : regarder la gestion des données de Célestin avec le + 150. 
    - [ ] Reconstruction `full MS` avec une observation / une TS complète (metrics + objectif de visu)
    - [X] Reconstruction `full MS` obtention des métriques pour un test complets.
    - [ ] Reconstruction seulement des bandes à `RGB-NIR` (10m resolution seulement.) 
    - [X] Pour chacun des cas vérifier que les métriques soient faites selon les manières suivantes:
        - [X] métriques spécifiques aux dates avec nuages synthétiques qui seront reconstruites
        - [X] métriques spécifiques aux dates sans nuages syn pour voir à quel point le modèle ne les altèrent pas.
        - [X] métriques OA pour avoir une représentation des perfs du modèle sur l'ensemble des dates.
        - [X] appliquer les mêmes métriques avec les masks de Célestin et les dates complètement cachées.
        - [ ] same avec seulement les pixels appartenant à des parcelles agricoles présentes dans le RPG.

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

### Annexe si temps
- [ ] Mettre l'affichage de config avec rich comme dans le repo UnCRtainTS
- [X] Comprendre dans le pairing de dates S2/S1 les index ne sont pas bons? => mauvaise idée de prendre l'index de la nouvelle liste, il aurait fallu pas refaire l'index mais mettre direct la valeur.
- [ ] Potentiellement refaire les fichiers hdf5 avec les TS de moins de 10 dates? 
- [ ] Explorer ce que fait le paramètre -mask dans la sélection des channels.
- [ ] Réparer la data augmentation.