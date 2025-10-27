# TODO LIST
### Aujourd'hui:
- [X] Vérifier l'aspect des masks nuages (d'origine) en sortie des dataloaders
- [X] Relancer les métriques
- [ ] Regarder l'état du training avec U-TILISE
- [ ] voir pourquoi les trainings prennent autant de temps sur jzellou / en local => comment faire pour accélérer le code. Serait-il mieux de tous passer rapidement sur lightning? 


run une boucle avec juste mae et sans torchmetrics: erreur arrive à 156
    => 'Info batch:', {'mgrs': ['30TYP'], 'mgrs25': ['30TYP_row-2_col-3'], 'window': ['0_0_256_256']}
seconde erreur:
'mgrs': ['30TYP'], 'mgrs25': ['30TYP_row-2_col-3'], 'window': ['0_0_256_256']
'mgrs': ['30TYP'], 'mgrs25': ['30TYP_row-4_col-3'], 'window': ['1280_512_256_256']
'mgrs': ['31UFP'], 'mgrs25': ['31UFP_row-3_col-3'], 'window': ['1792_256_256_256']
'mgrs': ['31UFP'], 'mgrs25': ['31UFP_row-4_col-2'], 'window': ['512_1280_256_256']
'mgrs': ['31UDR'], 'mgrs25': ['31UDR_row-4_col-4'], 'window': ['1536_1940_256_256']

- [ ] Travailler sur le géoréfécement des données et la sorties de prédictions durant une eval.
Relancer les trainings et les validations 

- [ ] Récupérer et mettre au propre les métriques calculés
    - [ ] ALL BANDS SAR
    - [ ] ALL BANDS
    - [ ] BGR NIR 
    - [ ] BGR NIR SAR
    - [ ] Anciens trainings JZAY 
- [ ] Sortir des métriques avec les modèles déjà entrainés sur les nouveaux jeu de 

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