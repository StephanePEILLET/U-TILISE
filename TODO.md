# TODO LIST
- [ ] Récupération et lancement de calcul de métriques sur jzellou
    - [X] training random_fully_masked => inference random_clouds
    - [X] training random_fully_masked => inferenced consecutive_fully_masked
  
    - [X] training consecutive_fully_masked => inference random_clouds
    - [X] training consecutive_fully_masked => inference random_fully_masked
    - [X] training consecutive_fully_masked => inf consecutive_fully_masked
  
    - [X] training loss occluded => inference random_clouds
    - [X] training loss_occluded => inference random_fully_masked
    - [X] training loss_occluded => infernce consecutive_fully_masked

- [ ] Training sur Jzay à faire:
    - [ ] Entrainement fait que sur les bandes RGB-NIR
    - [ ] Entrainement SAR ASC + DESC 
    - [ ] Réparer la partie test dans le code / vérifier pourquoi elle ne fonctionne pas sur jzay (automatiser pour faire le test sur les 3 cas de masking: random cloud / random_fully_masked / consecutive_fully_masked)

- [ ] Faire un notebook permettant de visualiser les attentions dans un notebook:
    - [ ] comprendre pourquoi je n’ai pas de matrices d’attention en sortie du modèle

- [ ] Coder les métriques par bandes pour voir quels sont les bandes les mieux reconstruites

- [ ] Commencer la rédaction du document permettant de justifier la non prise en compte des modèles U-TILISE et UnCRtainTS:
	- [ ] Partie U-TILISE:
        - [ ] Explication du modèle
        - [ ] Inférence et premiers résultats
	- [ ] Partie UnCRtainTS:
        - [ ] Explication du modèle
        - [ ] Inférence et premiers résultats

