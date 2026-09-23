-- ============================================================
-- Détection de fraude santé - Healthcare Provider Fraud Detection
-- Construction de la table agrégée au niveau du prestataire (Provider)
-- ============================================================

-- Aperçu de la table principale (1 ligne par prestataire, statut de fraude inclus)
SELECT * FROM Train LIMIT 10;

-- Granularité de chaque table : Train a une ligne par Provider (5410 = 5410),
-- alors que Train_Inpatientdata et Train_Outpatientdata ont plusieurs lignes
-- par Provider (plusieurs claims par prestataire). C'est ce qui dicte la
-- stratégie de jointure plus bas.
SELECT COUNT(*) AS nb_lignes_total,
       COUNT(DISTINCT Provider) AS nb_Provider_distinct
FROM Train;

SELECT COUNT(*) AS nb_lignes_total,
       COUNT(DISTINCT Provider) AS nb_Provider_distinct
FROM Train_Inpatientdata;

SELECT COUNT(*) AS nb_lignes_total,
       COUNT(DISTINCT Provider) AS nb_Provider_distinct
FROM Train_Outpatientdata;

-- Déséquilibre de classe : 506 fraudeurs sur 5410 prestataires (~9%)
SELECT PotentialFraud, COUNT(Provider) AS nb_providers
FROM Train
GROUP BY PotentialFraud;

-- Volume d'hospitalisation par prestataire (une seule table détaillée
-- jointe ici, pas de risque de fan-out)
SELECT
    Train.Provider,
    PotentialFraud,
    COUNT(Train_Inpatientdata.ClaimID) AS nb_hospit
FROM Train
JOIN Train_Inpatientdata
    ON Train.Provider = Train_Inpatientdata.Provider
GROUP BY Train.Provider, PotentialFraud
LIMIT 10;

-- Montants ambulatoires par prestataire
SELECT Train.Provider, Train.PotentialFraud,
       SUM(Train_Outpatientdata.InscClaimAmtReimbursed) AS total_rembourse_outpatient
FROM Train
JOIN Train_Outpatientdata
    ON Train.Provider = Train_Outpatientdata.Provider
GROUP BY Train.Provider, Train.PotentialFraud
LIMIT 10;

-- Contrôle de cohérence sur les fraudeurs connus
SELECT Train.Provider, Train.PotentialFraud,
       SUM(Train_Outpatientdata.InscClaimAmtReimbursed) AS total_rembourse_outpatient
FROM Train
JOIN Train_Outpatientdata
    ON Train.Provider = Train_Outpatientdata.Provider
WHERE Train.PotentialFraud = 'Yes'
GROUP BY Train.Provider, Train.PotentialFraud
LIMIT 5;


-- ------------------------------------------------------------
-- Piège du fan-out - NE PAS UTILISER, conservée à titre de contre-exemple
--
-- Joindre directement deux tables ayant chacune plusieurs lignes par
-- Provider (Train_Inpatientdata et Train_Outpatientdata) sans les agréger
-- au préalable provoque un produit local : chaque ligne inpatient d'un
-- prestataire rencontre chaque ligne outpatient de ce même prestataire,
-- ce qui gonfle artificiellement les COUNT et les SUM. Corrigée ci-dessous.
-- ------------------------------------------------------------
SELECT Train.Provider, Train.PotentialFraud,
       COUNT(DISTINCT Train_Inpatientdata.ClaimID) AS nb_hospit,
       SUM(Train_Outpatientdata.InscClaimAmtReimbursed) AS total_outpatient
FROM Train
LEFT JOIN Train_Inpatientdata
    ON Train.Provider = Train_Inpatientdata.Provider
LEFT JOIN Train_Outpatientdata
    ON Train.Provider = Train_Outpatientdata.Provider
GROUP BY Train.Provider, Train.PotentialFraud
LIMIT 10;


-- Correction : chaque table détaillée est agrégée séparément (une sous-
-- requête GROUP BY Provider par table) avant d'être jointe à Train - règle
-- générale dès que plusieurs tables à granularité fine doivent être combinées
SELECT
    Train.Provider,
    Train.PotentialFraud,
    inp_agg.nb_hospit,
    out_agg.total_outpatient
FROM Train
LEFT JOIN (
    SELECT
        Provider,
        COUNT(DISTINCT ClaimID) AS nb_hospit
    FROM Train_Inpatientdata
    GROUP BY Provider
) AS inp_agg
    ON Train.Provider = inp_agg.Provider
LEFT JOIN (
    SELECT
        Provider,
        SUM(InscClaimAmtReimbursed) AS total_outpatient
    FROM Train_Outpatientdata
    GROUP BY Provider
) AS out_agg
    ON Train.Provider = out_agg.Provider
ORDER BY nb_hospit DESC;


-- ============================================================
-- Requête finale : table complète pour la modélisation (9 variables)
-- Alimente directement les pipelines R (GLM, LASSO) et Python (XGBoost).
-- COALESCE(..., 0) : un NULL ici est un zéro structurel (un prestataire
-- 100% ambulatoire n'a simplement aucune ligne dans Train_Inpatientdata),
-- pas une donnée manquante.
-- ============================================================
SELECT
    t.Provider,
    t.PotentialFraud,
    COALESCE(inp.nb_hospit, 0) + COALESCE(out.nb_outpatient, 0) AS total_claims,
    COALESCE(inp.nb_hospit, 0) AS nb_hospit,
    COALESCE(out.nb_outpatient, 0) AS nb_outpatient,
    COALESCE(inp.total_reimb_inp, 0) + COALESCE(out.total_reimb_out, 0) AS total_reimbursed,
    COALESCE(inp.avg_reimb_inp, 0) AS avg_reimbursed_inp,
    COALESCE(out.avg_reimb_out, 0) AS avg_reimbursed_out,
    COALESCE(inp.max_reimb_inp, 0) AS max_reimbursed_inp,
    COALESCE(out.max_reimb_out, 0) AS max_reimbursed_out,
    COALESCE(inp.total_ded_inp, 0) + COALESCE(out.total_ded_out, 0) AS total_deductible
FROM Train t
LEFT JOIN (
    SELECT
        Provider,
        COUNT(DISTINCT ClaimID) AS nb_hospit,
        SUM(InscClaimAmtReimbursed) AS total_reimb_inp,
        AVG(InscClaimAmtReimbursed) AS avg_reimb_inp,
        MAX(InscClaimAmtReimbursed) AS max_reimb_inp,
        SUM(DeductibleAmtPaid) AS total_ded_inp
    FROM Train_Inpatientdata
    GROUP BY Provider
) inp ON t.Provider = inp.Provider
LEFT JOIN (
    SELECT
        Provider,
        COUNT(DISTINCT ClaimID) AS nb_outpatient,
        SUM(InscClaimAmtReimbursed) AS total_reimb_out,
        AVG(InscClaimAmtReimbursed) AS avg_reimb_out,
        MAX(InscClaimAmtReimbursed) AS max_reimb_out,
        SUM(DeductibleAmtPaid) AS total_ded_out
    FROM Train_Outpatientdata
    GROUP BY Provider
) out ON t.Provider = out.Provider;
