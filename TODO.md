# TODO LIST
### Aujourd'hui:
Lancer une inférence pour U-TILISE:
    - sortir les métriques en fonction des différents masquages
run une boucle juste avec les inférences
run une boucle avec les métrics (juste mae et sans torchmetrics)
run une boucle sans utiliser l'objet torchmetric pour voir si déjà cela passe..

Tâches à faire:
    - regarder où se trouve le dataset: CIRCA_CR_merged.hdf5 (LNV87/store_dai equipiers)


- [ ] Travailler sur le géoréfécement des données et la sorties de prédictions durant une eval.
Relancer les trainings et les validations 

- [ ] Récupérer et mettre au propre les métriques calculés
    - [ ] ALL BANDS SAR
    - [ ] ALL BANDS
    - [ ] BGR NIR 
    - [ ] BGR NIR SAR
    - [ ] Anciens trainings JZAY 
- [ ] Sortir des métriques avec les modèles déjà entrainés sur les nouveaux jeu de 
- [ ] voir pourquoi les trainings prennent autant de temps sur jzellou / en local => comment faire pour accélérer le code.
- [ ] relancer des trainings avec les cas de masquage sur des dates consécutives.
- [ ] Mettre en place le système d'imputation pour le training afin d'avoir une validation sur une TS complète



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
    - [ ] Faire une comparaison avec les trainings fait auparavant (cf nécessité du code de métriques)

---

### Annexe si temps
- [ ] Mettre l'affichage de config avec rich comme dans le repo UnCRtainTS
- [ ] Potentiellement refaire les fichiers hdf5 avec les TS de moins de 10 dates? 
- [ ] Réparer la data augmentation.