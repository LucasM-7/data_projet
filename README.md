# Détection de fraude santé : Healthcare Provider Fraud Detection

**Pipeline complet SQL → R → Python, du nettoyage de données brutes jusqu'à un modèle déployé en production.**

Projet construit sur le dataset Kaggle *Healthcare Provider Fraud Detection* (claims santé US), avec l'objectif de prédire, à partir du comportement agrégé d'un prestataire de soins, sa probabilité d'être frauduleux.

**Démo en ligne :** [dataprojet-twklxaurry8b9srgoe47dz.streamlit.app](https://dataprojet-twklxaurry8b9srgoe47dz.streamlit.app)

---

## Contexte et objectif

Un assureur santé ne peut pas auditer tous ses prestataires, auditer coûte cher, et laisser passer une fraude coûte plus cher encore. L'objectif de ce projet est double :

1. **Détecter** les prestataires au comportement de facturation atypique, à partir de leur activité agrégée (volume de claims, montants remboursés, part hospitalisation/ambulatoire, franchise).
2. **Décider** quand déclencher un audit, en tenant compte du coût réel d'un audit inutile face au coût d'une fraude non détectée, pas seulement de la performance statistique brute du modèle.

## Stack et rôle de chaque outil

| Étape | Outil | Rôle |
|---|---|---|
| Exploration & jointures | **SQL** (DBeaver + SQLite) | Jointure des 3 tables sources (`Train`, `Train_Inpatientdata`, `Train_Outpatientdata`) et agrégation au niveau prestataire : construction des 9 variables explicatives finales |
| Modélisation statistique | **R** (glmnet, caret, pROC, PRROC) | EDA, baseline GLM, LASSO avec validation croisée, diagnostic de multicolinéarité, calibration du seuil de décision |
| Modélisation ML & déploiement | **Python** (scikit-learn, XGBoost, SHAP, Streamlit) | Modèle de comparaison (XGBoost + SHAP), ré-implémentation du LASSO retenu, dashboard interactif |

## Méthodologie

- **Split stratifié 80/20** effectué *avant* tout preprocessing, pour éviter toute fuite de données (data leakage).
- **Seuil de décision calibré sur des prédictions Out-Of-Fold (OOF)** du train uniquement, jamais sur le test, qui reste sanctuarisé pour l'évaluation finale.
- Deux seuils comparés dans le dashboard final :
un seuil statistique (indice de Youden), qui traite faux positifs et faux négatifs à égalité ;
un seuil coût-sensible métier, calibré pour minimiser un coût estimé (300€ par audit inutile vs. montant remboursé en cas de fraude manquée). Recherché sur une grille resserrée près de zéro (pas de 0.002 sous 0.05, puis 0.05 au-delà), avec vérification que l'optimum trouvé (seuil = 0.019) n'est pas en bordure de grille , ce qui confirme qu'il s'agit d'un vrai minimum du coût total, pas d'un artefact des bornes testées.
- **Diagnostic de multicolinéarité** (matrice de corrélation + VIF) pour comprendre pourquoi le LASSO ne retient que 3 variables sur 9. Quatre redondances identifiées : une identité algébrique exacte (`total_claims` = `nb_hospit` + `nb_outpatient`), deux quasi-identités structurelles (`total_deductible` ≈ `nb_hospit` — franchise Medicare quasi fixe par séjour ; `avg_reimbursed_inp` ≈ `max_reimbursed_inp` — un seul séjour hospitalier pour la plupart des prestataires), et une forte corrélation (`avg_reimbursed_out`/`max_reimbursed_out`, 0.95, VIF > 29).

## Résultats (test set, 1081 prestataires jamais vus)

| Modèle | ROC-AUC | PR-AUC | Variables retenues |
|---|---|---|---|
| GLM baseline (2 variables, R) | 0.912 | — | `log_outpatient`, `nb_hospit` |
| LASSO (R, glmnet) | 0.929 | 0.674 | `total_reimbursed`, `total_claims`, `total_deductible` |
| **LASSO (Python, scikit-learn — modèle déployé)** | **0.930** | **0.705** | 3 |
| Elastic Net (R, glmnet) | 0.928 | 0.677 | `total_claims`, `nb_hospit`, `nb_outpatient`, `total_reimbursed`, `max_reimbursed_inp`, `total_deductible` |
| XGBoost (Python, comparaison) | 0.920 | 0.693 | 9 (top 3 en importance SHAP : `total_reimbursed`, `total_deductible`, `max_reimbursed_out`) |

**Modèle final retenu : LASSO.** Une courbe d'apprentissage sur XGBoost (PR-AUC en validation croisée selon la taille du train) montre une nette progression jusqu'à ~2770 lignes (0.65 → 0.68), puis un plafonnement au-delà, suggérant que le dataset a d'abord limité XGBoost sur les petites tailles, mais qu'avec le volume actuel on approche déjà du plafond atteignable avec ces 9 variables. Autrement dit, le potentiel non-linéaire supplémentaire de XGBoost semble limité ici, que ce soit par la nature essentiellement linéaire de la relation ou par la taille encore modeste de l'échantillon (~400 fraudeurs), les deux hypothèses ne s'excluent pas. Dans tous les cas, cela justifie le choix du LASSO, plus simple et plus interprétable, pour la version déployée.

Le LASSO a été implémenté deux fois (R avec `glmnet`, Python avec `scikit-learn`) pour comparer les deux écosystèmes ; le léger écart de performance entre les deux vient de la différence d'algorithme d'optimisation et de sélection du paramètre de régularisation (`lambda.1se` en validation croisée pour `glmnet`, vs. une grille de `C` pour `LogisticRegressionCV`), pas d'un bug. C'est la version Python qui est déployée dans le dashboard Streamlit.

Variables retenues par le LASSO (par poids décroissant) : `total_reimbursed`, `total_claims`, `total_deductible`.

## Rigueur méthodologique : erreurs identifiées et corrigées

Un projet de data science n'est pas linéaire, voici les erreurs rencontrées en cours de route et comment elles ont été corrigées, par souci de transparence :

1. **Fuite de données sur le seuil de décision.** La toute première version calibrait et évaluait le seuil sur l'intégralité du dataset, sans split train/test. → Corrigé par l'introduction d'un split stratifié strict et d'un seuil calibré uniquement sur des prédictions OOF du train.
2. **Seuil optimisé directement sur le test (exploration initiale).** Accepté comme simplification à ce stade exploratoire, mais identifié comme une fuite de données plus subtile, corrigé dans le pipeline final.
3. **Biais de construction d'une variable métier** (`cost_per_hospit`) : la première formule mélangeait un montant ambulatoire au numérateur avec un nombre d'hospitalisations au dénominateur, rendant le ratio non interprétable. Identifié et reformulé.
4. **Non-reproductibilité stricte malgré un `set.seed()` fixé** : une mise à jour de package a changé les résultats entre deux exécutions. Leçon retenue : figer les versions des packages (`renv::snapshot()` en R, `requirements.txt` versionné en Python) plutôt que de se reposer uniquement sur la seed.
5. **`vif()` sur le modèle complet plantait** (`there are aliased coefficients in the model`) : le GLM à 9 variables contient plusieurs quasi-identités exactes (voir ci-dessus), donc certains coefficients ne sont pas identifiables tant qu'on ne les a pas retirés. Corrigé en diagnostiquant d'abord avec `alias()`, puis en calculant le VIF uniquement sur le modèle réduit.

## Dashboard interactif

L'application Streamlit permet de simuler le profil d'un prestataire (nombre de claims, montants remboursés, franchise, etc.) et d'obtenir :
- la probabilité de fraude estimée par le modèle,
- la décision d'audit selon les deux logiques de seuil (statistique vs. coût-sensible),
- les métriques de performance du modèle sur le test set.

## Données

Dataset source : [Healthcare Provider Fraud Detection](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis) (Kaggle). Les fichiers de données ne sont pas inclus dans ce repo, à télécharger séparément sur Kaggle pour ré-exécuter les scripts `analysis/`.

## Structure du repo

```
data_projet/
├── app.py                       # Dashboard Streamlit
├── requirements.txt
├── README.md
├── streamlit_artifacts/         # Modèle LASSO, preprocessor, seuil (sérialisés)
└── analysis/
    ├── 01_sql/
    │   └── socle_fraude.sql     # Jointures et agrégation au niveau prestataire (9 variables)
    ├── 02_r/
    │   └── pipeline_fraude.R    # EDA, GLM, LASSO, Elastic Net, calibration du seuil
    └── 03_python/
        └── vigie_fraude.py      # LASSO (ré-implémentation), XGBoost, SHAP, seuil coût-sensible
```

## Pistes d'amélioration

- Exploiter les dates de réclamation (ClaimStartDxDate/ClaimEndDxDate), non utilisées dans la version actuelle, pour tester la robustesse du modèle dans le temps (split temporel plutôt qu'aléatoire, détection de dérive).
- Ajouter un renv/requirements figé pour garantir la reproductibilité stricte entre exécutions.
- Étendre le dashboard avec des profils pré-remplis issus du dataset réel, pour faciliter la démonstration.

---

*Projet réalisé par Lucas M. dans le cadre d'une préparation à des postes de data scientist en assurance santé / mutuelle.* d'une préparation à des postes de data scientist en assurance santé / mutuelle.*
