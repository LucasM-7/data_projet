# ============================================================
# Détection de fraude santé - Healthcare Provider Fraud Detection
# Pipeline R : exploration, modèle GLM/LASSO, calibration du seuil
#
# Structure :
#   Partie A - Exploration rapide (insurance_clean.csv, 2 variables)
#   Partie B - Pipeline final (df_lasso.csv, 9 variables issues de SQL)
#   Partie C - Feature engineering exploratoire (non intégré au modèle)
# ============================================================

# Reproductibilité stricte : versions des packages figées via renv
# (renv::init() puis renv::snapshot() après installation des dépendances,
# voir renv.lock)

library(dplyr)
library(ggplot2)
library(caret)
library(pROC)
library(PRROC)
library(glmnet)
library(corrplot)
library(car)
library(tidyverse)

set.seed(2026)


# ============================================================
# PARTIE A - Exploration initiale (2 variables)
# ============================================================

df <- read.csv("insurance_clean.csv")

df <- df %>%
  mutate(
    nb_hospit = ifelse(is.na(nb_hospit), 0, nb_hospit),
    total_outpatient = ifelse(is.na(total_outpatient), 0, total_outpatient),
    PotentialFraud = as.factor(PotentialFraud)
  )

summary(df)
colSums(is.na(df))

# Distributions en log1p (variables monétaires très asymétriques)
ggplot(df, aes(x = log1p(nb_hospit), fill = PotentialFraud)) +
  geom_density(alpha = 0.5) +
  labs(title = "Log(Nombre d'hospitalisation) selon fraude potentielle")

ggplot(df, aes(x = log1p(total_outpatient), fill = PotentialFraud)) +
  geom_density(alpha = 0.5) +
  labs(title = "Log(montant ambulatoire) selon fraude potentielle")

df %>%
  group_by(PotentialFraud) %>%
  summarise(
    med_hospit = median(nb_hospit),
    q3_hospit = quantile(nb_hospit, 0.75),
    med_outpatient = median(total_outpatient),
    q3_outpatient = quantile(total_outpatient, 0.75)
  )

df$log_total_outpatient <- log1p(df$total_outpatient)

ggplot(df, aes(x = PotentialFraud, y = log_total_outpatient, fill = PotentialFraud)) +
  geom_boxplot(alpha = 0.7) +
  labs(
    title = "Comparaison des montants ambulatoires (échelle log)",
    x = "Fraude potentielle",
    y = "Log(Total Outpatient + 1)"
  ) +
  theme_minimal()

# Outliers (Tukey) : conservés volontairement, un montant extrême est
# précisément le signal recherché dans ce contexte métier
seuilHaut <- 22615 + (1.5 * 21013)
sum(df$total_outpatient > seuilHaut)  # 601 / 5410 prestataires (~11%)

df$log_outpatient <- log1p(df$total_outpatient)
summary(df$log_outpatient)

df %>%
  group_by(PotentialFraud) %>%
  summarise(
    med_log = median(log_outpatient),
    q3_log = quantile(log_outpatient, 0.75),
    med_euro = expm1(median(log_outpatient)),
    q3_euro = expm1(quantile(log_outpatient, 0.75))
  )

# Split stratifié 80/20
train_index <- createDataPartition(df$PotentialFraud, p = 0.80, list = FALSE)
train_df <- df[train_index, ]
test_df <- df[-train_index, ]

nrow(train_df)  # 4329
nrow(test_df)   # 1081
prop.table(table(train_df$PotentialFraud))
prop.table(table(test_df$PotentialFraud))

# GLM baseline (2 variables)
model_glm <- glm(PotentialFraud ~ log_outpatient + nb_hospit, data = train_df, family = binomial)
summary(model_glm)
# AIC: 1824.6

exp(coef(model_glm))  # Odds Ratios
# nb_hospit      -> OR 1.074 (+7.4% de risque par hospitalisation)
# log_outpatient -> OR 1.105

# Évaluation au seuil par défaut (0.5)
test_prob <- predict(model_glm, newdata = test_df, type = "response")
test_pred <- ifelse(test_prob >= 0.5, "Yes", "No")
table(Prediction = test_pred, Realite = test_df$PotentialFraud)
# Précision 68.3%, recall 27.7% - seuil 0.5 inadapté à une classe minoritaire (~9%)

# Seuil optimal via indice de Youden (recherché ici sur le test, à titre
# exploratoire - corrigé en Partie B avec une approche Out-Of-Fold)
roc_score <- roc(test_df$PotentialFraud, test_prob)
auc(roc_score)  # 0.9119
plot(roc_score, main = "Courbe ROC - Baseline GLM")

coords(roc_score, "best", ret = c("threshold", "sensitivity", "specificity"))
# threshold = 0.0524 | sensitivity = 0.9406 | specificity = 0.7622

test_pred_opt <- ifelse(test_prob >= 0.05240409, "Yes", "No")
table(Prediction = test_pred_opt, Realite = test_df$PotentialFraud)
# Recall porté à 94%, au prix de faux positifs supplémentaires (arbitrage
# precision/recall repris en Partie B via un seuil coût-sensible)


# ============================================================
# PARTIE B - Pipeline final (9 variables, requête SQL)
# ============================================================

df <- read.csv("df_lasso.csv")

df <- df %>%
  mutate(
    nb_hospit = ifelse(is.na(nb_hospit), 0, nb_hospit),
    nb_outpatient = ifelse(is.na(nb_outpatient), 0, nb_outpatient)
  )

df$PotentialFraud <- ifelse(df$PotentialFraud == "Yes", 1, 0)
df$PotentialFraud <- factor(df$PotentialFraud, levels = c(0, 1))

# Split stratifié AVANT tout preprocessing, pour éviter le data leakage
train_index <- createDataPartition(df$PotentialFraud, p = 0.80, list = FALSE)
train_df <- df[train_index, ]
test_df <- df[-train_index, ]

mean(train_df$PotentialFraud == 1)  # 0.094
mean(test_df$PotentialFraud == 1)   # 0.093

# Export des identifiants Provider, pour rejouer le même split côté Python
write.csv(data.frame(Provider = train_df$Provider), "train_providers.csv", row.names = FALSE)
write.csv(data.frame(Provider = test_df$Provider), "test_providers.csv", row.names = FALSE)

# Preprocessing : log1p sur les variables monétaires, appliqué séparément
# train/test (aucune statistique calibrée sur l'ensemble du dataset)
cols_log <- c("total_reimbursed", "avg_reimbursed_inp", "avg_reimbursed_out",
              "max_reimbursed_inp", "max_reimbursed_out")

train_prep <- train_df
test_prep <- test_df
for (col in cols_log) {
  if (col %in% colnames(train_prep)) {
    train_prep[[col]] <- log1p(train_prep[[col]])
    test_prep[[col]] <- log1p(test_prep[[col]])
  }
}

# Diagnostic de multicolinéarité (corrélation + VIF)
vars_num <- train_prep %>% select(-Provider, -PotentialFraud)
cor_matrix <- cor(vars_num, use = "complete.obs")
round(cor_matrix, 2)

corrplot(cor_matrix, method = "color", type = "upper",
         addCoef.col = "black", tl.col = "black", tl.srt = 45,
         number.cex = 0.7,
         title = "Corrélation entre les 9 variables explicatives (Train, post-log)",
         mar = c(0, 0, 1, 0))

glm_full <- glm(PotentialFraud ~ . - Provider, data = train_prep, family = binomial)
vif(glm_full)
glm_full_reduced <- update(glm_full, . ~ . - nb_outpatient - total_deductible - max_reimbursed_inp)
vif(glm_full_reduced)
alias(glm_full)

# Deux redondances identifiées parmi les 9 variables construites en SQL :
# - identité algébrique exacte : total_claims = nb_hospit + nb_outpatient
# - redondance structurelle : avg_reimbursed_out et max_reimbursed_out
#   coïncident pour tout prestataire n'ayant qu'un seul claim ambulatoire
#   (corrélation observée 0.95, VIF > 29)
# Ces deux redondances expliquent pourquoi le LASSO ne retient que 3
# variables sur 9.

# GLM baseline enrichi (3 variables, sur Train uniquement)
glm_baseline <- glm(PotentialFraud ~ total_claims + nb_hospit + avg_reimbursed_out,
                    data = train_prep,
                    family = binomial(link = "logit"))

exp(cbind(OR = coef(glm_baseline), confint.default(glm_baseline)))
# avg_reimbursed_out a un OR < 1 (effet négatif) - à surveiller, cf. le
# même type d'effet de structure que cost_per_hospit (Partie C)

X_test <- model.matrix(PotentialFraud ~ . - Provider - 1, data = test_prep)
y_test <- as.numeric(as.character(test_prep$PotentialFraud))

glm_test_probs <- predict(glm_baseline, newdata = test_prep, type = "response")
auc(roc(y_test, glm_test_probs))
pr.curve(scores.class0 = glm_test_probs[y_test == 1], scores.class1 = glm_test_probs[y_test == 0])$auc.integral

# LASSO : cross-validation, alpha = 1, optimisation sur l'AUC
X_train <- model.matrix(PotentialFraud ~ . - Provider - 1, data = train_prep)
y_train <- as.numeric(as.character(train_prep$PotentialFraud))

set.seed(2026)
foldid <- sample(rep(1:10, length.out = nrow(X_train)))

cv_lasso <- cv.glmnet(x = X_train, y = y_train, family = "binomial",
                      alpha = 1, type.measure = "auc", keep = TRUE,
                      foldid = foldid)

best_lambda_idx <- which(cv_lasso$lambda == cv_lasso$lambda.1se)
oof_preds <- cv_lasso$fit.preval[, best_lambda_idx]

# Elastic Net (alpha = 0.5) en comparaison directe, même foldid
cv_enet <- cv.glmnet(x = X_train, y = y_train, family = "binomial",
                     alpha = 0.5, type.measure = "auc", keep = TRUE,
                     foldid = foldid)

coef(cv_enet, s = cv_enet$lambda.1se)

best_lambda_idx_enet <- which(cv_enet$lambda == cv_enet$lambda.1se)
oof_preds_enet <- cv_enet$fit.preval[, best_lambda_idx_enet]
oof_probs_enet <- 1 / (1 + exp(-oof_preds_enet))

roc_train_oof_enet <- roc(y_train, oof_probs_enet)
optimal_threshold_enet <- as.numeric(coords(roc_train_oof_enet, "best", ret = "threshold", best.method = "youden")$threshold[1])

test_probs_enet <- as.vector(predict(cv_enet, newx = X_test, s = cv_enet$lambda.1se, type = "response"))
test_preds_binary_enet <- ifelse(test_probs_enet >= optimal_threshold_enet, 1, 0)

cm_enet <- confusionMatrix(factor(test_preds_binary_enet), factor(y_test), positive = "1")
auc_roc_enet <- auc(roc(y_test, test_probs_enet))
auc_pr_enet <- pr.curve(scores.class0 = test_probs_enet[y_test == 1],
                        scores.class1 = test_probs_enet[y_test == 0])$auc.integral
brier_score_enet <- mean((test_probs_enet - y_test)^2)

# Calibration du seuil LASSO sur des prédictions Out-Of-Fold (OOF) du train
# uniquement - jamais sur le test, qui reste sanctuarisé pour l'évaluation
# finale. C'est la correction méthodologique clé de ce pipeline (voir notes
# de fin de script).
oof_probs <- 1 / (1 + exp(-oof_preds))

roc_train_oof <- roc(y_train, oof_probs)
optimal_threshold <- as.numeric(coords(roc_train_oof, "best", ret = "threshold", best.method = "youden")$threshold[1])

cv_lasso$lambda.1se       # 0.06686
optimal_threshold         # 0.1013

# Évaluation finale sur le test sanctuarisé
test_probs <- as.vector(predict(cv_lasso, newx = X_test, s = cv_lasso$lambda.1se, type = "response"))
test_preds_binary <- ifelse(test_probs >= optimal_threshold, 1, 0)

cm <- confusionMatrix(factor(test_preds_binary), factor(y_test), positive = "1")
roc_test <- roc(y_test, test_probs)
auc_roc <- auc(roc_test)

# PR-AUC plus pertinent que ROC-AUC ici (classe positive minoritaire, ~9%)
pr_obj <- pr.curve(scores.class0 = test_probs[y_test == 1],
                   scores.class1 = test_probs[y_test == 0],
                   curve = TRUE)
auc_pr <- pr_obj$auc.integral

# Brier Score : qualité de calibration des probabilités, pas seulement
# leur capacité à discriminer
brier_score <- mean((test_probs - y_test)^2)

auc_roc      # 0.9286
auc_pr       # 0.674
brier_score  # 0.0616
cm$table
cm$byClass[c("Sensitivity", "Specificity", "Precision", "F1")]

# Tableau comparatif final
comparaison <- data.frame(
  Modele = c("GLM baseline", "LASSO", "Elastic Net", "XGBoost"),
  ROC_AUC = c(NA, round(auc_roc, 4), round(auc_roc_enet, 4), 0.9139),
  PR_AUC  = c(NA, round(auc_pr, 4), round(auc_pr_enet, 4), 0.6468),
  Nb_variables_retenues = c(NA, 3, 6, NA)
)
print(comparaison)

# Interprétabilité : coefficients non nuls du LASSO (lambda.1se), en Odds Ratio
lasso_coefs <- coef(cv_lasso, s = cv_lasso$lambda.1se)
df_coefs <- data.frame(
  Feature = rownames(lasso_coefs),
  Coefficient = as.vector(lasso_coefs)
) %>%
  filter(Coefficient != 0 & Feature != "(Intercept)") %>%
  mutate(Odds_Ratio = exp(Coefficient)) %>%
  arrange(desc(abs(Coefficient)))
print(df_coefs)
# total_reimbursed, total_claims, total_deductible (par poids décroissant)

# Calibration (probabilités prédites vs. proportion réelle de fraude par
# quintile) et stabilité de l'AUC en cross-validation
test_eval_df <- data.frame(prob = test_probs, actual = y_test) %>%
  mutate(bin = ntile(prob, 5)) %>%
  group_by(bin) %>%
  summarise(mean_pred = mean(prob), mean_actual = mean(actual), count = n())
print(test_eval_df)

plot(test_eval_df$mean_pred, test_eval_df$mean_actual, type = "b", col = "blue", pch = 19,
     xlab = "Probabilités prédites moyennes", ylab = "Proportion réelle de fraudes",
     main = "Courbe de calibration - LASSO Test Set")
abline(0, 1, lty = 2, col = "red")

max(cv_lasso$cvm)                    # AUC moyenne en CV : 0.9414
cv_lasso$cvsd[best_lambda_idx]       # écart-type : 0.0039 (performance stable entre folds)


# ============================================================
# PARTIE C - Feature engineering exploratoire (non intégré au modèle final)
# ============================================================
# Variable testée : coût moyen par hospitalisation. Statut : non tranché,
# à valider dans une prochaine itération ou à retirer.
#
# Première version abandonnée : mélangeait total_outpatient (ambulatoire)
# au numérateur avec nb_hospit au dénominateur - deux grandeurs de nature
# différente, ratio non interprétable.
# df$cost_per_hospit <- (df$total_outpatient / (df$nb_hospit + 1))  # FAUX

# Version révisée : montant hospitalier / nombre d'hospitalisations (+1
# pour éviter la division par 0)
df <- df %>%
  mutate(
    total_inpatient = coalesce(avg_reimbursed_inp * nb_hospit, 0),
    cost_per_hospit = total_inpatient / (nb_hospit + 1)
  )
summary(df$cost_per_hospit)

# Sur l'ancienne version du ratio, les fraudeurs semblaient avoir un coût
# par hospitalisation plus bas (contre-intuitif) - probablement un effet
# de structure lié à une forte activité d'hospitalisation. À revalider
# avec la version corrigée avant toute utilisation dans un modèle.


# ============================================================
# Notes méthodologiques
# ============================================================
# 1. Fuite de données sur le seuil de décision : la première itération de
#    l'analyse LASSO évaluait directement sur l'intégralité du dataset,
#    sans split. Corrigé par un split stratifié strict + seuil calibré
#    sur des prédictions Out-Of-Fold du train, appliqué gelé sur le test.
#
# 2. Seuil optimisé sur le test (Partie A, exploration) : acceptable au
#    stade exploratoire, mais fuite de données plus subtile si utilisée
#    pour la décision finale. Corrigée en Partie B (approche OOF).
#
# 3. cost_per_hospit : première formule mélangeant deux catégories de
#    montants différentes, ratio non interprétable. Reformulée en Partie C.
#
# 4. Résultats LASSO non stables entre deux exécutions malgré set.seed()
#    fixé : cv.glmnet consomme le générateur aléatoire pour ses folds, et
#    une mise à jour de package a changé ce comportement interne. Un seed
#    seul ne garantit pas la reproductibilité stricte dans le temps, d'où
#    le choix de figer les versions via renv (voir renv.lock).
