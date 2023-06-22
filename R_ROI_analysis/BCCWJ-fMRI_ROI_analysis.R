#BCCWJ-fMRI analysis
#Predictors: 5-gram, LSTM, surp.RNNGs, dis.RNNGs
#(baseline: word_freq, word_rate, sentid, sentpos, six + head movement parameters) 
#fMRI data: BCCWJ-fMRI

######packages#########
library(lme4)
library(lmerTest)
#####################

#road data
data <- read.csv('../data/Results-ts/all/ts.csv',header=TRUE, sep="\t")

# scaling baseline predictors
data$word_rate = scale(data$word_rate)
data$word_length = scale(data$word_length)
data$word_freq = scale(data$word_freq)
data$sentid = scale(data$sentid)
data$sentpos = scale(data$sentpos)

#scaling head movement predictors
data$dx = scale(data$dx)
data$dy = scale(data$dy)
data$dz = scale(data$dz)
data$rx = scale(data$rx)
data$ry = scale(data$ry)
data$rz = scale(data$rz)

data$subject_number = as.factor(data$subject_number)
data$section_number = as.factor(data$section_number)

#scaling surprisals and distance
data$surp.ngram_five = scale(data$surp.ngram_five)
data$LSTM = scale(data$surp.LSTM)
data$RNNG_LC = scale(data$surp.RNNG_LC)
data$RNNG_TD = scale(data$surp.RNNG_TD)
data$dis_RNNG_LC = scale(data$dis_RNNG_LC)
data$dis_RNNG_TD = scale(data$dis_RNNG_TD)

#fitting baseline
control = lmerControl(optCtrl = list(maxfun=100000))

########### (1) Frontal_Inf_Oper_L ###################
baseline_Frontal_Inf_Oper_L = lmer(Frontal_Inf_Oper_L ~ sentpos + sentid + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), data,REML = FALSE)

#Remove data points beyond three standard deviations
dataTrim_Frontal_Inf_Oper_L = data[scale(resid(baseline_Frontal_Inf_Oper_L)) < 3.0,]
nrow(data) - nrow(dataTrim_Frontal_Inf_Oper_L) 
(nrow(data) - nrow(dataTrim_Frontal_Inf_Oper_L)) / nrow(data) 

#fitting the model using trimmed data
baseline_Frontal_Inf_Oper_L = lmer(Frontal_Inf_Oper_L ~ sentpos + sentid + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Oper_L, REML = FALSE)

####### models #########
#ngram-five
surp.ngram_five_Frontal_Inf_Oper_L = lmer(Frontal_Inf_Oper_L ~ surp.ngram_five + sentpos + sentid + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Oper_L, REML = FALSE)

#ngram-five + LSTM 
surp.ngram_five_LSTM_Frontal_Inf_Oper_L = lmer(Frontal_Inf_Oper_L ~ surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Oper_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_TD
surp.ngram_five_LSTM_RNNG_TD_Frontal_Inf_Oper_L = lmer(Frontal_Inf_Oper_L ~ surp.RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Oper_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_LC
surp.ngram_five_LSTM_RNNG_LC_Frontal_Inf_Oper_L = lmer(Frontal_Inf_Oper_L ~ surp.RNNG_LC + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate+ dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Oper_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_TD + surp.RNNG_LC
surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Oper_L = lmer(Frontal_Inf_Oper_L ~ surp.RNNG_LC + surp.RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Oper_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_LC
surp.ngram_five_LSTM_dis_RNNG_LC_Frontal_Inf_Oper_L = lmer(Frontal_Inf_Oper_L ~ dis_RNNG_LC + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Oper_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_TD
surp.ngram_five_LSTM_dis_RNNG_TD_Frontal_Inf_Oper_L = lmer(Frontal_Inf_Oper_L ~ dis_RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Oper_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_TD + dis_RNNG_LC
surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Frontal_Inf_Oper_L = lmer(Frontal_Inf_Oper_L ~ dis_RNNG_LC + dis_RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Oper_L, REML = FALSE)
#Model Comparisions 
#baseline < surp.ngram_five < LSTM < surp.RNNG_TD < surp.RNNG_LC
anova(baseline_Frontal_Inf_Oper_L, surp.ngram_five_Frontal_Inf_Oper_L, surp.ngram_five_LSTM_Frontal_Inf_Oper_L,surp.ngram_five_LSTM_RNNG_TD_Frontal_Inf_Oper_L,surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Oper_L)
#surp_RNNG_LC < surp.RNNG_TD
anova(surp.ngram_five_LSTM_RNNG_LC_Frontal_Inf_Oper_L, surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Oper_L)
#LSTM < surp.RNNG_TD
anova(surp.ngram_five_LSTM_Frontal_Inf_Oper_L, surp.ngram_five_LSTM_RNNG_TD_Frontal_Inf_Oper_L) 
#LSTM <surp.RNNG_LC
anova(surp.ngram_five_LSTM_Frontal_Inf_Oper_L, surp.ngram_five_LSTM_RNNG_LC_Frontal_Inf_Oper_L)
#LSTM < dis_RNNG_LC
anova(surp.ngram_five_LSTM_Frontal_Inf_Oper_L, surp.ngram_five_LSTM_dis_RNNG_LC_Frontal_Inf_Oper_L) 
#LSTM < dis_RNNG_TD
anova(surp.ngram_five_LSTM_Frontal_Inf_Oper_L, surp.ngram_five_LSTM_dis_RNNG_TD_Frontal_Inf_Oper_L) 
#dis_RNNG_TD < dis_RNNG_LC
anova(surp.ngram_five_LSTM_dis_RNNG_TD_Frontal_Inf_Oper_L, surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Frontal_Inf_Oper_L) 
#dis_RNNG_LC < dis_RNNG_TD
anova(surp.ngram_five_LSTM_dis_RNNG_LC_Frontal_Inf_Oper_L, surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Frontal_Inf_Oper_L) 

########### (2) Frontal_Inf_Tri_L ###################
baseline_Frontal_Inf_Tri_L = lmer(Frontal_Inf_Tri_L ~ sentpos + sentid + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), data, REML = FALSE)

#Remove data points beyond three standard deviations
dataTrim_Frontal_Inf_Tri_L = data[scale(resid(baseline_Frontal_Inf_Tri_L)) < 3.0 , ]
nrow(data) - nrow(dataTrim_Frontal_Inf_Tri_L) 
(nrow(data) - nrow(dataTrim_Frontal_Inf_Tri_L)) / nrow(data)

#fitting the model using trimmed data
baseline_Frontal_Inf_Tri_L = lmer(Frontal_Inf_Tri_L ~ sentpos + sentid + word_freq + word_length + word_rate +  dx + dy + dz + rx + ry+ rz  + (1 | subject_number), dataTrim_Frontal_Inf_Tri_L, REML = FALSE)

####### models #########
#ngram-five
surp.ngram_five_Frontal_Inf_Tri_L = lmer(Frontal_Inf_Tri_L ~ surp.ngram_five + sentpos + sentid + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Tri_L, REML = FALSE)

#ngram-five + LSTM 
surp.ngram_five_LSTM_Frontal_Inf_Tri_L = lmer(Frontal_Inf_Tri_L ~ surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Tri_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_TD
surp.ngram_five_LSTM_RNNG_TD_Frontal_Inf_Tri_L = lmer(Frontal_Inf_Tri_L ~ surp.RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length +word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Tri_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_LC
surp.ngram_five_LSTM_RNNG_LC_Frontal_Inf_Tri_L = lmer(Frontal_Inf_Tri_L ~ surp.RNNG_LC + surp.LSTM + surp.ngram_five + sentid + sentpos   + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Tri_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_TD + surp.RNNG_LC
surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Tri_L = lmer(Frontal_Inf_Tri_L ~ surp.RNNG_LC + surp.RNNG_TD + surp.LSTM  + surp.ngram_five + sentid + sentpos  + word_freq  + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Tri_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_LC
surp.ngram_five_LSTM_dis_RNNG_LC_Frontal_Inf_Tri_L = lmer(Frontal_Inf_Tri_L ~ dis_RNNG_LC + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate+ dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Tri_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_TD
surp.ngram_five_LSTM_dis_RNNG_TD_Frontal_Inf_Tri_L = lmer(Frontal_Inf_Tri_L ~ dis_RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Tri_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_TD + dis_RNNG_LC
surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Frontal_Inf_Tri_L = lmer(Frontal_Inf_Tri_L ~ dis_RNNG_LC + dis_RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Tri_L, REML = FALSE)

#Model Comparisions 
#baseline < surp.ngram_five < LSTM < surp.RNNG_TD < surp.RNNG_LC
anova(baseline_Frontal_Inf_Tri_L, surp.ngram_five_Frontal_Inf_Tri_L, surp.ngram_five_LSTM_Frontal_Inf_Tri_L, surp.ngram_five_LSTM_RNNG_TD_Frontal_Inf_Tri_L,surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Tri_L)
#surp.RNNG_LC < surp.RNNG_TD
anova(surp.ngram_five_LSTM_RNNG_LC_Frontal_Inf_Tri_L, surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Tri_L)
#LSTM < surp.RNNG_TD
anova(surp.ngram_five_LSTM_Frontal_Inf_Tri_L, surp.ngram_five_LSTM_RNNG_TD_Frontal_Inf_Tri_L) 
#LSTM <surp.RNNG_LC
anova(surp.ngram_five_LSTM_Frontal_Inf_Tri_L, surp.ngram_five_LSTM_RNNG_LC_Frontal_Inf_Tri_L)
#LSTM < dis_RNNG_LC
anova(surp.ngram_five_LSTM_Frontal_Inf_Tri_L, surp.ngram_five_LSTM_dis_RNNG_LC_Frontal_Inf_Tri_L)
#LSTM < dis_RNNG_TD
anova(surp.ngram_five_LSTM_Frontal_Inf_Tri_L, surp.ngram_five_LSTM_dis_RNNG_TD_Frontal_Inf_Tri_L)
#dis_RNNG_TD < dis_RNNG_LC
anova(surp.ngram_five_LSTM_dis_RNNG_TD_Frontal_Inf_Tri_L, surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Frontal_Inf_Tri_L)
#dis_RNNG_LC < dis_RNNG_TD
anova(surp.ngram_five_LSTM_dis_RNNG_LC_Frontal_Inf_Tri_L, surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Frontal_Inf_Tri_L)

########### (3) Frontal_Inf_Orb_L ###################
baseline_Frontal_Inf_Orb_L = lmer(Frontal_Inf_Orb_L ~ sentpos + sentid + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), data, REML = FALSE)

#Remove data points beyond three standard deviations
dataTrim_Frontal_Inf_Orb_L = data[scale(resid(baseline_Frontal_Inf_Orb_L)) < 3.0 , ]
nrow(data) - nrow(dataTrim_Frontal_Inf_Orb_L) 
(nrow(data) - nrow(dataTrim_Frontal_Inf_Orb_L)) / nrow(data)

#fitting the model using trimmed data
baseline_Frontal_Inf_Orb_L = lmer(Frontal_Inf_Orb_L ~ sentpos + sentid + word_freq + word_length + word_rate +  dx + dy + dz + rx + ry+ rz  + (1 | subject_number), dataTrim_Frontal_Inf_Orb_L, REML = FALSE)

####### models #########
#ngram-five
surp.ngram_five_Frontal_Inf_Orb_L = lmer(Frontal_Inf_Orb_L ~ surp.ngram_five + sentpos + sentid + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Orb_L, REML = FALSE)

#ngram-five + LSTM 
surp.ngram_five_LSTM_Frontal_Inf_Orb_L = lmer(Frontal_Inf_Orb_L ~ surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Orb_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_TD
surp.ngram_five_LSTM_RNNG_TD_Frontal_Inf_Orb_L = lmer(Frontal_Inf_Orb_L ~ surp.RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length +word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Orb_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_LC
surp.ngram_five_LSTM_RNNG_LC_Frontal_Inf_Orb_L = lmer(Frontal_Inf_Orb_L ~ surp.RNNG_LC + surp.LSTM + surp.ngram_five + sentid + sentpos   + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Orb_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_TD + surp.RNNG_LC
surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Orb_L = lmer(Frontal_Inf_Orb_L ~ surp.RNNG_LC + surp.RNNG_TD + surp.LSTM  + surp.ngram_five + sentid + sentpos  + word_freq  + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Orb_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_LC
surp.ngram_five_LSTM_dis_RNNG_LC_Frontal_Inf_Orb_L = lmer(Frontal_Inf_Orb_L ~ dis_RNNG_LC + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate+ dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Orb_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_TD
surp.ngram_five_LSTM_dis_RNNG_TD_Frontal_Inf_Orb_L = lmer(Frontal_Inf_Orb_L ~ dis_RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Orb_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_TD_LC
surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Frontal_Inf_Orb_L = lmer(Frontal_Inf_Orb_L ~ dis_RNNG_LC + dis_RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Frontal_Inf_Orb_L, REML = FALSE)

#Model Comparisions 
#baseline < surp.ngram_five < LSTM < surp.RNNG_TD < surp.RNNG_LC
anova(baseline_Frontal_Inf_Orb_L, surp.ngram_five_Frontal_Inf_Orb_L, surp.ngram_five_LSTM_Frontal_Inf_Orb_L, surp.ngram_five_LSTM_RNNG_TD_Frontal_Inf_Orb_L,surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Orb_L)
#surp_RNNG_LC < surp.RNNG_TD
anova(surp.ngram_five_LSTM_RNNG_LC_Frontal_Inf_Orb_L, surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Orb_L)
#LSTM < surp.RNNG_TD
anova(surp.ngram_five_LSTM_Frontal_Inf_Orb_L, surp.ngram_five_LSTM_RNNG_TD_Frontal_Inf_Orb_L) 
#LSTM <surp.RNNG_LC
anova(surp.ngram_five_LSTM_Frontal_Inf_Orb_L, surp.ngram_five_LSTM_RNNG_LC_Frontal_Inf_Orb_L)
#LSTM < dis_RNNG_LC
anova(surp.ngram_five_LSTM_Frontal_Inf_Orb_L, surp.ngram_five_LSTM_dis_RNNG_LC_Frontal_Inf_Orb_L)
#LSTM < dis_RNNG_TD
anova(surp.ngram_five_LSTM_Frontal_Inf_Orb_L, surp.ngram_five_LSTM_dis_RNNG_TD_Frontal_Inf_Orb_L)
# dis_RNNG_TD < dis_RNNG_LC
anova(surp.ngram_five_LSTM_dis_RNNG_TD_Frontal_Inf_Orb_L, surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Frontal_Inf_Orb_L)
# dis_RNNG_LC < dis_RNNG_TD
anova(surp.ngram_five_LSTM_dis_RNNG_LC_Frontal_Inf_Orb_L, surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Frontal_Inf_Orb_L)

########### (4)  Parietal_Inf_L  ###################
baseline_Parietal_Inf_L = lmer(Parietal_Inf_L ~ sentpos + sentid + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), data, REML = FALSE)

#Remove data points beyond three standard deviations
dataTrim_Parietal_Inf_L = data[scale(resid(baseline_Parietal_Inf_L)) < 3.0 , ]
nrow(data) - nrow(dataTrim_Parietal_Inf_L) 
(nrow(data) - nrow(dataTrim_Parietal_Inf_L)) / nrow(data)

#fitting the model using trimmed data
baseline_Parietal_Inf_L = lmer(Parietal_Inf_L ~ sentpos + sentid + word_freq + word_length + word_rate +  dx + dy + dz + rx + ry+ rz  + (1 | subject_number), dataTrim_Parietal_Inf_L, REML = FALSE)

####### models #########
#ngram-five
surp.ngram_five_Parietal_Inf_L = lmer(Parietal_Inf_L ~ surp.ngram_five + sentpos + sentid + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Parietal_Inf_L, REML = FALSE)

#ngram-five + LSTM 
surp.ngram_five_LSTM_Parietal_Inf_L = lmer(Parietal_Inf_L ~ surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Parietal_Inf_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_TD
surp.ngram_five_LSTM_RNNG_TD_Parietal_Inf_L = lmer(Parietal_Inf_L ~ surp.RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length +word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Parietal_Inf_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_LC
surp.ngram_five_LSTM_RNNG_LC_Parietal_Inf_L = lmer(Parietal_Inf_L ~ surp.RNNG_LC + surp.LSTM + surp.ngram_five + sentid + sentpos   + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Parietal_Inf_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_TD + surp.RNNG_LC
surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Parietal_Inf_L = lmer(Parietal_Inf_L ~ surp.RNNG_LC + surp.RNNG_TD + surp.LSTM  + surp.ngram_five + sentid + sentpos  + word_freq  + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Parietal_Inf_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_LC
surp.ngram_five_LSTM_dis_RNNG_LC_Parietal_Inf_L = lmer(Parietal_Inf_L ~ dis_RNNG_LC + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate+ dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Parietal_Inf_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_TD
surp.ngram_five_LSTM_dis_RNNG_TD_Parietal_Inf_L = lmer(Parietal_Inf_L ~ dis_RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Parietal_Inf_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_TD + dis_RNNG_LC
surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Parietal_Inf_L = lmer(Parietal_Inf_L ~ dis_RNNG_LC + dis_RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Parietal_Inf_L, REML = FALSE)

#Model Comparisions 
#baseline < surp.ngram_five < LSTM < RNNG_TD < RNNG_LC
anova(baseline_Parietal_Inf_L, surp.ngram_five_Parietal_Inf_L, surp.ngram_five_LSTM_Parietal_Inf_L, surp.ngram_five_LSTM_RNNG_TD_Parietal_Inf_L,surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Parietal_Inf_L)
#surp_RNNG_LC < surp_RNNG_TD
anova(surp.ngram_five_LSTM_RNNG_LC_Parietal_Inf_L, surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Parietal_Inf_L)
#LSTM < surp.RNNG_TD
anova(surp.ngram_five_LSTM_Parietal_Inf_L, surp.ngram_five_LSTM_RNNG_TD_Parietal_Inf_L) 
#LSTM <surp.RNNG_LC
anova(surp.ngram_five_LSTM_Parietal_Inf_L, surp.ngram_five_LSTM_RNNG_LC_Parietal_Inf_L)
#LSTM < dis_RNNG_LC
anova(surp.ngram_five_LSTM_Parietal_Inf_L, surp.ngram_five_LSTM_dis_RNNG_LC_Parietal_Inf_L)
#LSTM < dis_RNNG_TD
anova(surp.ngram_five_LSTM_Parietal_Inf_L, surp.ngram_five_LSTM_dis_RNNG_TD_Parietal_Inf_L)
# dis_RNNG_TD < dis_RNNG_LC
anova(surp.ngram_five_LSTM_dis_RNNG_TD_Parietal_Inf_L, surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Parietal_Inf_L)
# dis_RNNG_LC < dis_RNNG_TD
anova(surp.ngram_five_LSTM_dis_RNNG_LC_Parietal_Inf_L, surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Parietal_Inf_L)

########### (5)  Angular_L  ###################
baseline_Angular_L = lmer(Angular_L ~ sentpos + sentid + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), data, REML = FALSE)

#Remove data points beyond three standard deviations
dataTrim_Angular_L = data[scale(resid(baseline_Angular_L)) < 3.0 , ]
nrow(data) - nrow(dataTrim_Angular_L) 
(nrow(data) - nrow(dataTrim_Angular_L)) / nrow(data)

#fitting the model using trimmed data
baseline_Angular_L = lmer(Angular_L ~ sentpos + sentid + word_freq + word_length + word_rate +  dx + dy + dz + rx + ry+ rz  + (1 | subject_number), dataTrim_Angular_L, REML = FALSE)

####### models #########
#ngram-five
surp.ngram_five_Angular_L = lmer(Angular_L ~ surp.ngram_five + sentpos + sentid + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Angular_L, REML = FALSE)

#ngram-five + LSTM 
surp.ngram_five_LSTM_Angular_L = lmer(Angular_L ~ surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Angular_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_TD
surp.ngram_five_LSTM_RNNG_TD_Angular_L = lmer(Angular_L ~ surp.RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length +word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Angular_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_LC
surp.ngram_five_LSTM_RNNG_LC_Angular_L = lmer(Angular_L ~ surp.RNNG_LC + surp.LSTM + surp.ngram_five + sentid + sentpos   + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Angular_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_TD + surp.RNNG_LC
surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Angular_L = lmer(Angular_L ~ surp.RNNG_LC + surp.RNNG_TD + surp.LSTM  + surp.ngram_five + sentid + sentpos  + word_freq  + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Angular_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_LC
surp.ngram_five_LSTM_dis_RNNG_LC_Angular_L = lmer(Angular_L ~ dis_RNNG_LC + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate+ dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Angular_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_TD
surp.ngram_five_LSTM_dis_RNNG_TD_Angular_L = lmer(Angular_L ~ dis_RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Angular_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_TD + dis_RNNG_LC
surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Angular_L = lmer(Angular_L ~ dis_RNNG_LC + dis_RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Angular_L, REML = FALSE)

#Model Comparisions 
#baseline < surp.ngram_five < LSTM < RNNG_TD < RNNG_LC
anova(baseline_Angular_L, surp.ngram_five_Angular_L, surp.ngram_five_LSTM_Angular_L, surp.ngram_five_LSTM_RNNG_TD_Angular_L,surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Angular_L)
#surp_RNNG_LC < surp_RNNG_TD
anova(surp.ngram_five_LSTM_RNNG_LC_Angular_L, surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Angular_L)
#LSTM < surp.RNNG_TD
anova(surp.ngram_five_LSTM_Angular_L, surp.ngram_five_LSTM_RNNG_TD_Angular_L) 
#LSTM <surp.RNNG_LC
anova(surp.ngram_five_LSTM_Angular_L, surp.ngram_five_LSTM_RNNG_LC_Angular_L)
#LSTM < dis_RNNG_LC
anova(surp.ngram_five_LSTM_Angular_L, surp.ngram_five_LSTM_dis_RNNG_LC_Angular_L)
#LSTM < dis_RNNG_TD
anova(surp.ngram_five_LSTM_Angular_L, surp.ngram_five_LSTM_dis_RNNG_TD_Angular_L)
# dis_RNNG_TD < dis_RNNG_LC
anova(surp.ngram_five_LSTM_dis_RNNG_TD_Angular_L, surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Angular_L)
# dis_RNNG_LC < dis_RNNG_TD
anova(surp.ngram_five_LSTM_dis_RNNG_LC_Angular_L, surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Angular_L)

########### (6)  Temporal_Sup_L  ###################
baseline_Temporal_Sup_L = lmer(Temporal_Sup_L ~ sentpos + sentid + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), data, REML = FALSE)

#Remove data points beyond three standard deviations
dataTrim_Temporal_Sup_L = data[scale(resid(baseline_Temporal_Sup_L)) < 3.0 , ]
nrow(data) - nrow(dataTrim_Temporal_Sup_L) 
(nrow(data) - nrow(dataTrim_Temporal_Sup_L)) / nrow(data)

#fitting the model using trimmed data
baseline_Temporal_Sup_L = lmer(Temporal_Sup_L ~ sentpos + sentid + word_freq + word_length + word_rate +  dx + dy + dz + rx + ry+ rz  + (1 | subject_number), dataTrim_Temporal_Sup_L, REML = FALSE)

####### models #########
#ngram-five
surp.ngram_five_Temporal_Sup_L = lmer(Temporal_Sup_L ~ surp.ngram_five + sentpos + sentid + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Sup_L, REML = FALSE)

#ngram-five + LSTM 
surp.ngram_five_LSTM_Temporal_Sup_L = lmer(Temporal_Sup_L ~ surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Sup_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_TD
surp.ngram_five_LSTM_RNNG_TD_Temporal_Sup_L = lmer(Temporal_Sup_L ~ surp.RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length +word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Sup_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_LC
surp.ngram_five_LSTM_RNNG_LC_Temporal_Sup_L = lmer(Temporal_Sup_L ~ surp.RNNG_LC + surp.LSTM + surp.ngram_five + sentid + sentpos   + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Sup_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_TD + surp.RNNG_LC
surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Temporal_Sup_L = lmer(Temporal_Sup_L ~ surp.RNNG_LC + surp.RNNG_TD + surp.LSTM  + surp.ngram_five + sentid + sentpos  + word_freq  + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Sup_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_LC
surp.ngram_five_LSTM_dis_RNNG_LC_Temporal_Sup_L = lmer(Temporal_Sup_L ~ dis_RNNG_LC + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate+ dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Sup_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_TD
surp.ngram_five_LSTM_dis_RNNG_TD_Temporal_Sup_L = lmer(Temporal_Sup_L ~ dis_RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Sup_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_TD + dis_RNNG_LC
surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Temporal_Sup_L = lmer(Temporal_Sup_L ~ dis_RNNG_LC + dis_RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Sup_L, REML = FALSE)

#Model Comparisions 
#baseline < surp.ngram_five < LSTM < RNNG_TD < RNNG_LC
anova(baseline_Temporal_Sup_L, surp.ngram_five_Temporal_Sup_L, surp.ngram_five_LSTM_Temporal_Sup_L, surp.ngram_five_LSTM_RNNG_TD_Temporal_Sup_L,surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Temporal_Sup_L)
#surp_RNNG_LC < surp_RNNG_TD
anova(surp.ngram_five_LSTM_RNNG_LC_Temporal_Sup_L, surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Temporal_Sup_L)
#LSTM < surp.RNNG_TD
anova(surp.ngram_five_LSTM_Temporal_Sup_L, surp.ngram_five_LSTM_RNNG_TD_Temporal_Sup_L) 
#LSTM <surp.RNNG_LC
anova(surp.ngram_five_LSTM_Temporal_Sup_L, surp.ngram_five_LSTM_RNNG_LC_Temporal_Sup_L)
#LSTM < dis_RNNG_LC
anova(surp.ngram_five_LSTM_Temporal_Sup_L, surp.ngram_five_LSTM_dis_RNNG_LC_Temporal_Sup_L)
#LSTM < dis_RNNG_TD
anova(surp.ngram_five_LSTM_Temporal_Sup_L, surp.ngram_five_LSTM_dis_RNNG_TD_Temporal_Sup_L)
# dis_RNNG_TD < dis_RNNG_LC
anova(surp.ngram_five_LSTM_dis_RNNG_TD_Temporal_Sup_L, surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Temporal_Sup_L)
# dis_RNNG_LC < dis_RNNG_TD
anova(surp.ngram_five_LSTM_dis_RNNG_LC_Temporal_Sup_L, surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Temporal_Sup_L)

########### (7)  Temporal_Pole_Sup_L  ###################
baseline_Temporal_Pole_Sup_L = lmer(Temporal_Pole_Sup_L ~ sentpos + sentid + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), data, REML = FALSE)

#Remove data points beyond three standard deviations
dataTrim_Temporal_Pole_Sup_L = data[scale(resid(baseline_Temporal_Pole_Sup_L)) < 3.0 , ]
nrow(data) - nrow(dataTrim_Temporal_Pole_Sup_L) 
(nrow(data) - nrow(dataTrim_Temporal_Pole_Sup_L)) / nrow(data)

#fitting the model using trimmed data
baseline_Temporal_Pole_Sup_L = lmer(Temporal_Pole_Sup_L ~ sentpos + sentid + word_freq + word_length + word_rate +  dx + dy + dz + rx + ry+ rz  + (1 | subject_number), dataTrim_Temporal_Pole_Sup_L, REML = FALSE)

####### models #########
#ngram-five
surp.ngram_five_Temporal_Pole_Sup_L = lmer(Temporal_Pole_Sup_L ~ surp.ngram_five + sentpos + sentid + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Pole_Sup_L, REML = FALSE)

#ngram-five + LSTM 
surp.ngram_five_LSTM_Temporal_Pole_Sup_L = lmer(Temporal_Pole_Sup_L ~ surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Pole_Sup_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_TD
surp.ngram_five_LSTM_RNNG_TD_Temporal_Pole_Sup_L = lmer(Temporal_Pole_Sup_L ~ surp.RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length +word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Pole_Sup_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_LC
surp.ngram_five_LSTM_RNNG_LC_Temporal_Pole_Sup_L = lmer(Temporal_Pole_Sup_L ~ surp.RNNG_LC + surp.LSTM + surp.ngram_five + sentid + sentpos   + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Pole_Sup_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_TD + surp.RNNG_LC
surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Temporal_Pole_Sup_L = lmer(Temporal_Pole_Sup_L ~ surp.RNNG_LC + surp.RNNG_TD + surp.LSTM  + surp.ngram_five + sentid + sentpos  + word_freq  + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Pole_Sup_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_LC
surp.ngram_five_LSTM_dis_RNNG_LC_Temporal_Pole_Sup_L = lmer(Temporal_Pole_Sup_L ~ dis_RNNG_LC + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate+ dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Pole_Sup_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_TD
surp.ngram_five_LSTM_dis_RNNG_TD_Temporal_Pole_Sup_L = lmer(Temporal_Pole_Sup_L ~ dis_RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Pole_Sup_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_TD + dis_RNNG_LC
surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Temporal_Pole_Sup_L = lmer(Temporal_Pole_Sup_L ~ dis_RNNG_LC + dis_RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Pole_Sup_L, REML = FALSE)

#Model Comparisions 
#baseline < surp.ngram_five < LSTM < RNNG_TD < RNNG_LC
anova(baseline_Temporal_Pole_Sup_L, surp.ngram_five_Temporal_Pole_Sup_L, surp.ngram_five_LSTM_Temporal_Pole_Sup_L, surp.ngram_five_LSTM_RNNG_TD_Temporal_Pole_Sup_L,surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Temporal_Pole_Sup_L)
#surp_RNNG_LC < surp_RNNG_TD
anova(surp.ngram_five_LSTM_RNNG_LC_Temporal_Pole_Sup_L, surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Temporal_Pole_Sup_L)
#LSTM < surp.RNNG_TD
anova(surp.ngram_five_LSTM_Temporal_Pole_Sup_L, surp.ngram_five_LSTM_RNNG_TD_Temporal_Pole_Sup_L) 
#LSTM <surp.RNNG_LC
anova(surp.ngram_five_LSTM_Temporal_Pole_Sup_L, surp.ngram_five_LSTM_RNNG_LC_Temporal_Pole_Sup_L)
#LSTM < dis_RNNG_LC
anova(surp.ngram_five_LSTM_Temporal_Pole_Sup_L, surp.ngram_five_LSTM_dis_RNNG_LC_Temporal_Pole_Sup_L)
#LSTM < dis_RNNG_TD
anova(surp.ngram_five_LSTM_Temporal_Pole_Sup_L, surp.ngram_five_LSTM_dis_RNNG_TD_Temporal_Pole_Sup_L)
# dis_RNNG_TD < dis_RNNG_LC
anova(surp.ngram_five_LSTM_dis_RNNG_TD_Temporal_Pole_Sup_L, surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Temporal_Pole_Sup_L)
# dis_RNNG_LC < dis_RNNG_TD
anova(surp.ngram_five_LSTM_dis_RNNG_LC_Temporal_Pole_Sup_L, surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Temporal_Pole_Sup_L)


########### (8) Temporal_Pole_Mid_L ###################
baseline_Temporal_Pole_Mid_L = lmer(Temporal_Pole_Mid_L ~ sentpos + sentid + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), data, REML = FALSE)

#Remove data points beyond three standard deviations
dataTrim_Temporal_Pole_Mid_L = data[scale(resid(baseline_Temporal_Pole_Mid_L)) < 3.0 , ]
nrow(data) - nrow(dataTrim_Temporal_Pole_Mid_L) 
(nrow(data) - nrow(dataTrim_Temporal_Pole_Mid_L)) / nrow(data)

#fitting the model using trimmed data
baseline_Temporal_Pole_Mid_L = lmer(Temporal_Pole_Mid_L ~ sentpos + sentid + word_freq + word_length + word_rate +  dx + dy + dz + rx + ry+ rz  + (1 | subject_number), dataTrim_Temporal_Pole_Mid_L, REML = FALSE)

####### models #########
#ngram-five
surp.ngram_five_Temporal_Pole_Mid_L = lmer(Temporal_Pole_Mid_L ~ surp.ngram_five + sentpos + sentid + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Pole_Mid_L, REML = FALSE)

#ngram-five + LSTM 
surp.ngram_five_LSTM_Temporal_Pole_Mid_L = lmer(Temporal_Pole_Mid_L ~ surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Pole_Mid_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_TD
surp.ngram_five_LSTM_RNNG_TD_Temporal_Pole_Mid_L = lmer(Temporal_Pole_Mid_L ~ surp.RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length +word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Pole_Mid_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_LC
surp.ngram_five_LSTM_RNNG_LC_Temporal_Pole_Mid_L = lmer(Temporal_Pole_Mid_L ~ surp.RNNG_LC + surp.LSTM + surp.ngram_five + sentid + sentpos   + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Pole_Mid_L, REML = FALSE)

#ngram-five + LSTM + surp.RNNG_TD + surp.RNNG_LC
surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Temporal_Pole_Mid_L = lmer(Temporal_Pole_Mid_L ~ surp.RNNG_LC + surp.RNNG_TD + surp.LSTM  + surp.ngram_five + sentid + sentpos  + word_freq  + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Pole_Mid_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_LC
surp.ngram_five_LSTM_dis_RNNG_LC_Temporal_Pole_Mid_L = lmer(Temporal_Pole_Mid_L ~ dis_RNNG_LC + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate+ dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Pole_Mid_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_TD
surp.ngram_five_LSTM_dis_RNNG_TD_Temporal_Pole_Mid_L = lmer(Temporal_Pole_Mid_L ~ dis_RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Pole_Mid_L, REML = FALSE)

#ngram-five + LSTM + dis_RNNG_TD + dis_RNNG_LC
surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Temporal_Pole_Mid_L = lmer(Temporal_Pole_Mid_L ~ dis_RNNG_LC + dis_RNNG_TD + surp.LSTM + surp.ngram_five + sentid + sentpos + word_freq + word_length + word_rate + dx + dy + dz + rx + ry+ rz + (1 | subject_number), dataTrim_Temporal_Pole_Mid_L, REML = FALSE)

#Model Comparisions 
#baseline < surp.ngram_five < LSTM < RNNG_TD < RNNG_LC
anova(baseline_Temporal_Pole_Mid_L, surp.ngram_five_Temporal_Pole_Mid_L, surp.ngram_five_LSTM_Temporal_Pole_Mid_L, surp.ngram_five_LSTM_RNNG_TD_Temporal_Pole_Mid_L,surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Temporal_Pole_Mid_L)
#surp_RNNG_LC < surp_RNNG_TD
anova(surp.ngram_five_LSTM_RNNG_LC_Temporal_Pole_Mid_L, surp.ngram_five_LSTM_RNNG_TD_RNNG_LC_Temporal_Pole_Mid_L)
#LSTM < surp.RNNG_TD
anova(surp.ngram_five_LSTM_Temporal_Pole_Mid_L, surp.ngram_five_LSTM_RNNG_TD_Temporal_Pole_Mid_L) 
#LSTM <surp.RNNG_LC
anova(surp.ngram_five_LSTM_Temporal_Pole_Mid_L, surp.ngram_five_LSTM_RNNG_LC_Temporal_Pole_Mid_L)
#LSTM < dis_RNNG_LC
anova(surp.ngram_five_LSTM_Temporal_Pole_Mid_L, surp.ngram_five_LSTM_dis_RNNG_LC_Temporal_Pole_Mid_L)
#LSTM < dis_RNNG_TD
anova(surp.ngram_five_LSTM_Temporal_Pole_Mid_L, surp.ngram_five_LSTM_dis_RNNG_TD_Temporal_Pole_Mid_L)
# dis_RNNG_TD < dis_RNNG_LC
anova(surp.ngram_five_LSTM_dis_RNNG_TD_Temporal_Pole_Mid_L, surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Temporal_Pole_Mid_L)
# dis_RNNG_LC < dis_RNNG_TD
anova(surp.ngram_five_LSTM_dis_RNNG_LC_Temporal_Pole_Mid_L, surp.ngram_five_LSTM_dis_RNNG_TD_dis_RNNG_LC_Temporal_Pole_Mid_L)

######################
#Bonferroi correction
##Threshold = a/N where a = 0.05, N = the number of tests 
##e.g., 0.05/8 (the number of ROIs) = 0.00625