library(dplyr)
library(data.table)
library(moonBook)
library(ggplot2)
library(caret)
library(precrec)

train <- fread("data/train.csv", data.table = FALSE)
val <- fread("data/val.csv", data.table = FALSE)

train$Sex_f <- as.factor(train$Sex)
train$Multiple_f <- as.factor(train$Multiple)
train$Piecemeal_f <- as.factor(train$Piecemeal)
train$Location_f <- as.factor(train$Location)
train$Differentiation_f <- as.factor(train$Differentiation)
train$SM2_f <- as.factor(train$SM2)
train$DM_f <- as.factor(train$DM)
train$HTN_f <- as.factor(train$HTN)
train$LC_f <- as.factor(train$LC)
train$CKD_f <- as.factor(train$CKD)
train$Aspirin_f <- as.factor(train$Aspirin)
train$P2Y12RA_f <- as.factor(train$P2Y12RA)
train$Warfarin_f <- as.factor(train$Warfarin)
train$DOAC_f <- as.factor(train$DOAC)
train$Anticoagulant_f <- as.factor(train$Anticoagulant)
train$Cilostazol_f <- as.factor(train$Cilostazol)
train$NSAIDs_f <- as.factor(train$NSAIDs)
train$Interruption_Bridge_or_Replace_f <- as.factor(train$Interruption_Bridge_or_Replace)
train$Hb_drop_over2_f <- as.factor(train$Hb_drop_over2)

train$PEB <- as.factor(train$PEB)


val$Sex_f <- as.factor(val$Sex)
val$Multiple_f <- as.factor(val$Multiple)
val$Piecemeal_f <- as.factor(val$Piecemeal)
val$Location_f <- as.factor(val$Location)
val$Differentiation_f <- as.factor(val$Differentiation)
val$SM2_f <- as.factor(val$SM2)
val$DM_f <- as.factor(val$DM)
val$HTN_f <- as.factor(val$HTN)
val$LC_f <- as.factor(val$LC)
val$CKD_f <- as.factor(val$CKD)
val$Aspirin_f <- as.factor(val$Aspirin)
val$P2Y12RA_f <- as.factor(val$P2Y12RA)
val$Warfarin_f <- as.factor(val$Warfarin)
val$DOAC_f <- as.factor(val$DOAC)
val$Anticoagulant_f <- as.factor(val$Anticoagulant)
val$Cilostazol_f <- as.factor(val$Cilostazol)
val$NSAIDs_f <- as.factor(val$NSAIDs)
val$Interruption_Bridge_or_Replace_f <- as.factor(val$Interruption_Bridge_or_Replace)
val$Hb_drop_over2_f <- as.factor(val$Hb_drop_over2)

val$PEB <- as.factor(val$PEB)


#Variables
x_var <- c("Sex_f", "Age",
           "DM_f", "HTN_f", "LC_f", "CKD_f", 
           "Aspirin_f", "P2Y12RA_f", "Warfarin_f", "DOAC_f", "Cilostazol_f", "NSAIDs_f", "Interruption_Bridge_or_Replace_f",
           "Multiple_f", "Location_f", "Differentiation_f", "Size_tumor", "Piecemeal_f",
           "Albumin_preESD", "INR_preESD") 

my_formula <- as.formula(paste0("PEB ~ ", paste0(x_var, collapse = " + ")))
tr_glm <- glm(my_formula, family = binomial, data = train)
extractOR(tr_glm)
vif(tr_glm)
summary(tr_glm)

val_pred <- predict(tr_glm, newdata = val, type="response")
val_pred <- as.data.frame(val_pred) %>% rename(PROB = val_pred)
summary(val_pred)

val_pred <- cbind(val, val_pred)
val_pred$NTILE <- ntile(val_pred$PROB, 10)

tmp <- summary(tr_glm)
tmp <- as.data.frame(tmp$coefficients)
colnames(tmp)[4] <- "PVAL"
tmp$VAR <- rownames(tmp)
tmp2 <- filter(tmp, PVAL < 0.05)
tmp2$VAR
#[1] "(Intercept)"      "Sex_f2"           "Age"              "HTN_f1"           "CKD_f1"           "P2Y12RA_f1"       "Location_f2"      "Size_tumor"      
#[9] "Hb_drop_over2_f1"
val_pred$PROB2 <- filter(tmp2, VAR == "(Intercept)")$Estimate + 
  filter(tmp2, VAR == "Sex_f2")$Estimate*ifelse(val_pred$Sex==2, 1, 0) +
  filter(tmp2, VAR == "Age")$Estimate*log(val_pred$Age) +
  filter(tmp2, VAR == "HTN_f1")$Estimate*ifelse(val_pred$HTN==1, 1, 0) +
  filter(tmp2, VAR == "CKD_f1")$Estimate*ifelse(val_pred$CKD==1, 1, 0) +
  filter(tmp2, VAR == "P2Y12RA_f1")$Estimate*ifelse(val_pred$P2Y12RA==1, 1, 0) +
  filter(tmp2, VAR == "Location_f2")$Estimate*ifelse(val_pred$Location==2, 1, 0) +
  filter(tmp2, VAR == "Size_tumor")$Estimate*log(val_pred$Size_tumor)

val_pred$PROB2 <- 1/(1+exp(-val_pred$PROB2))


# Calculate ROC and Precision-Recall curves
sscurves <- evalmod(scores = val_pred$PROB2, labels = val_pred$PEB)
sscurves
autoplot(sscurves)
