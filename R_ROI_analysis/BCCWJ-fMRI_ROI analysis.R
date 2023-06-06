#BCCWJ-fMRI analysis
#Predictors: 5-gram, LSTM, RNNGs, 
#(baseline: word_freq, word_rate, sentid, sentpos, head movement parameters) 
#fMRI data: BCCWJ-fMRI

######packages#########
library(tidyr)
library(lme4)
library(lmerTest)
library(lmtest)
library(RLRsim)
library(coefplot)
library(ggplot2)
library("gridExtra")
library(modelsummary)
library(stargazer)
palette()
#####################

#road the big file, data from BCCWJ-fMRI and predictors
data <- read.csv('./analyses/R_ROI_analysis/ts_20230330.csv',header=TRUE, sep="\t")

# scaling baseline predictors
data$word_rate = scale(data$word_rate)
data$word_length = scale(data$word_length)
data$word_freq = scale(data$word_freq)
data$sentid = scale(data$sentid)
data$sentpos = scale(data$sentpos)

# scaling head movement predictors
data$dx = scale(data$dx)
data$dy = scale(data$dy)
data$dz = scale(data$dz)

data$rx = scale(data$rx)
data$ry = scale(data$ry)
data$rz = scale(data$rz)

data$subject_number = as.factor(data$subject_number)
data$section_number = as.factor(data$section_number)

# scaling surprisals and distance
data$ngram_five = scale(data$ngram_five)
data$LSTM_seed_1 = scale(data$LSTM_seed_1)
data$RNNG_LC_1_4 = scale(data$RNNG_LC_1_4)
data$RNNG_TD_2_10 = scale(data$RNNG_TD_2_10)
data$dis_RNNG_LC_1_4 = scale(data$dis_RNNG_LC_1_4)
data$dis_RNNG_TD_2_10 = scale(data$dis_RNNG_TD_2_10)

hist(data$word_rate)
hist(data$word_freq)
hist(data$ngram_five)
hist(data$word_length)
hist(data$sentid)
hist(data$sentpos)

hist(data$LSTM_seed_1)
hist(data$RNNG_TD_2_10)
hist(data$RNNG_LC_1_4)
hist(data$dis_RNNG_LC_1_4)
hist(data$dis_RNNG_TD_2_10)

# fitting baseline
control = lmerControl(optCtrl = list(maxfun=100000))

########### (1) Frontal_Inf_Oper_L ###################
baseline_Frontal_Inf_Oper_L = lmer(
  Frontal_Inf_Oper_L ~sentpos + sentid + word_freq + word_length + word_rate + 
                      dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  data,
  REML = FALSE
)
summary(baseline_Frontal_Inf_Oper_L)
coefplot(baseline_Frontal_Inf_Oper_L)

# get rid of outliers
dataTrim_Frontal_Inf_Oper_L = data[scale(resid(baseline_Frontal_Inf_Oper_L)) < 3.0 , ]
nrow(data) - nrow(dataTrim_Frontal_Inf_Oper_L) 
(nrow(data) - nrow(dataTrim_Frontal_Inf_Oper_L)) / nrow(data) 

# fitting the model using trimmed data
baseline_Frontal_Inf_Oper_L = lmer(
  Frontal_Inf_Oper_L ~ sentpos + sentid + word_freq + word_length+ word_rate + 
                       dx + dy + dz + rx + ry+ rz + (1 | subject_number),
  dataTrim_Frontal_Inf_Oper_L,
  REML = FALSE
)
summary(baseline_Frontal_Inf_Oper_L)
coefplot(baseline_Frontal_Inf_Oper_L)


####### models #########
# ngram-five
ngram_five_Frontal_Inf_Oper_L = lmer(
  Frontal_Inf_Oper_L ~  ngram_five + 
                        sentpos + sentid + word_freq + word_length + word_rate + 
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Frontal_Inf_Oper_L,
  REML = FALSE
)
summary(ngram_five_Frontal_Inf_Oper_L)
coefplot(ngram_five_Frontal_Inf_Oper_L)
anova(baseline_Frontal_Inf_Oper_L,ngram_five_Frontal_Inf_Oper_L)  

#LSTM 
LSTM_Frontal_Inf_Oper_L = lmer(
  Frontal_Inf_Oper_L ~ LSTM_seed_1 + ngram_five  +  
                      sentid + sentpos  + word_freq + word_length  + word_rate +　 
                      dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Frontal_Inf_Oper_L,
  REML = FALSE
)
summary(LSTM_Frontal_Inf_Oper_L)
coefplot(LSTM_Frontal_Inf_Oper_L)
anova(ngram_five_Frontal_Inf_Oper_L,LSTM_Frontal_Inf_Oper_L) #5-gram<LSTM

#surp_RNNG_TD
RNNG_TD_Frontal_Inf_Oper_L = lmer(
  Frontal_Inf_Oper_L ~ RNNG_TD_2_10 + LSTM_seed_1 + ngram_five +  
                        sentid + sentpos + word_freq + word_length + word_rate + 
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Frontal_Inf_Oper_L,
  REML = FALSE
)
summary(RNNG_TD_Frontal_Inf_Oper_L)
coefplot(RNNG_TD_Frontal_Inf_Oper_L)
anova(LSTM_Frontal_Inf_Oper_L,RNNG_TD_Frontal_Inf_Oper_L) #LSTM <surp_RNNG_TD

#surp_RNNG_LC
RNNG_LC_Frontal_Inf_Oper_L = lmer(
  Frontal_Inf_Oper_L ~ RNNG_LC_1_4 + LSTM_seed_1 + ngram_five +  
                        sentid + sentpos + word_freq +  word_length + word_rate+　 
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Frontal_Inf_Oper_L,
  REML = FALSE
)
summary(RNNG_LC_Frontal_Inf_Oper_L)
coefplot(RNNG_LC_Frontal_Inf_Oper_L)
anova(LSTM_Frontal_Inf_Oper_L,RNNG_LC_Frontal_Inf_Oper_L) #LSTM <surp_RNNG_LC

#LSTM + surp_RNNG_TD + surp_RNNG_LC
LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Oper_L = lmer(
  Frontal_Inf_Oper_L ~ RNNG_LC_1_4 + RNNG_TD_2_10 + LSTM_seed_1  + ngram_five +
                        sentid + sentpos + word_freq + word_length + word_rate + 
                        dx + dy + dz + rx + ry+ rz + (1  |subject_number),
  dataTrim_Frontal_Inf_Oper_L,
  REML = FALSE
)
summary(LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Oper_L)
coefplot(LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Oper_L)

#baseline < ngram_five < LSTM < surp_RNNG_TD < surp_RNNG_LC
anova(baseline_Frontal_Inf_Oper_L,ngram_five_Frontal_Inf_Oper_L,
      LSTM_Frontal_Inf_Oper_L,RNNG_TD_Frontal_Inf_Oper_L,
      LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Oper_L)
anova(RNNG_LC_Frontal_Inf_Oper_L,
      LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Oper_L) #surp_RNNG_LC < surp_RNNG_TD

#RNNG_LC_dis
dis_RNNG_LC_Frontal_Inf_Oper_L = lmer(
  Frontal_Inf_Oper_L ~  dis_RNNG_LC_1_4 + LSTM_seed_1 + ngram_five +  
                        sentid + sentpos + word_freq +  word_length + word_rate+
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Frontal_Inf_Oper_L,
  REML = FALSE
)
summary(dis_RNNG_LC_Frontal_Inf_Oper_L)
coefplot(dis_RNNG_LC_Frontal_Inf_Oper_L)
anova(LSTM_Frontal_Inf_Oper_L,dis_RNNG_LC_Frontal_Inf_Oper_L) #LSTM < dis_RNNG_LC

#RNNG_TD_dis
dis_RNNG_TD_Frontal_Inf_Oper_L = lmer(
  Frontal_Inf_Oper_L ~ dis_RNNG_TD_2_10 + LSTM_seed_1 + ngram_five +  
                       sentid + sentpos + word_freq +  word_length + word_rate+
                       dx + dy + dz + rx + ry+ rz + (1   |subject_number),
  dataTrim_Frontal_Inf_Oper_L,
  REML = FALSE
)

summary(dis_RNNG_TD_Frontal_Inf_Oper_L)
coefplot(dis_RNNG_TD_Frontal_Inf_Oper_L)
anova(LSTM_Frontal_Inf_Oper_L,dis_RNNG_TD_Frontal_Inf_Oper_L) #LSTM < dis_RNNG_TD

#dis_RNNG_LC_TD
dis_RNNG_TD_LC_Frontal_Inf_Oper_L = lmer(
  Frontal_Inf_Oper_L ~  dis_RNNG_LC_1_4 + dis_RNNG_TD_2_10 + LSTM_seed_1 + ngram_five +  
                        sentid + sentpos + word_freq +  word_length + word_rate+　 
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Frontal_Inf_Oper_L,
  REML = FALSE
)
summary(dis_RNNG_TD_LC_Frontal_Inf_Oper_L)
coefplot(dis_RNNG_TD_LC_Frontal_Inf_Oper_L)
anova(dis_RNNG_TD_Frontal_Inf_Oper_L,dis_RNNG_TD_LC_Frontal_Inf_Oper_L) #dis_RNNG_LC>dis_RNNG_TD
anova(dis_RNNG_LC_Frontal_Inf_Oper_L,dis_RNNG_TD_LC_Frontal_Inf_Oper_L) #dis_RNNG_LC<dis_RNNG_TD

########### (2) Frontal_Inf_Tri_L ###################
baseline_Frontal_Inf_Tri_L = lmer(
  Frontal_Inf_Tri_L ~ sentpos + sentid + word_freq + word_length +word_rate + 
                      dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  data,
  REML = FALSE
)
summary(baseline_Frontal_Inf_Tri_L)
coefplot(baseline_Frontal_Inf_Tri_L)


dataTrim_Frontal_Inf_Tri_L = data[scale(resid(baseline_Frontal_Inf_Tri_L)) < 3.0 , ]
nrow(data) - nrow(dataTrim_Frontal_Inf_Tri_L) 　
(nrow(data) - nrow(dataTrim_Frontal_Inf_Tri_L)) / nrow(data)　


baseline_Frontal_Inf_Tri_L = lmer(
  Frontal_Inf_Tri_L ~  sentpos + sentid + word_freq + word_length + word_rate + 
                       dx + dy + dz + rx + ry+ rz  + (1 |subject_number),
  dataTrim_Frontal_Inf_Tri_L,
  REML = FALSE
)
summary(baseline_Frontal_Inf_Tri_L)
coefplot(baseline_Frontal_Inf_Tri_L)


####### models #########
# ngram-five
ngram_five_Frontal_Inf_Tri_L = lmer(
  Frontal_Inf_Tri_L ~  ngram_five + 
                        sentpos + sentid + word_freq + word_length +word_rate + 
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Frontal_Inf_Tri_L,
  REML = FALSE
)
summary(ngram_five_Frontal_Inf_Tri_L)
coefplot(ngram_five_Frontal_Inf_Tri_L)
anova(baseline_Frontal_Inf_Tri_L,ngram_five_Frontal_Inf_Tri_L)

# LSTM 
LSTM_Frontal_Inf_Tri_L = lmer(
  Frontal_Inf_Tri_L ~ LSTM_seed_1 + ngram_five  +  
                      sentid + sentpos + word_freq  + word_length + word_rate +
                      dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Frontal_Inf_Tri_L,
  REML = FALSE
)
summary(LSTM_Frontal_Inf_Tri_L)
coefplot(LSTM_Frontal_Inf_Tri_L)
anova(ngram_five_Frontal_Inf_Tri_L,LSTM_Frontal_Inf_Tri_L)


#surp_RNNG_TD
RNNG_TD_Frontal_Inf_Tri_L = lmer(
  Frontal_Inf_Tri_L ~ RNNG_TD_2_10 + LSTM_seed_1 + ngram_five +  
    sentid + sentpos + word_freq + word_length +word_rate  +　 
    dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Frontal_Inf_Tri_L,
  REML = FALSE
)
summary(RNNG_TD_Frontal_Inf_Tri_L)
coefplot(RNNG_TD_Frontal_Inf_Tri_L)

#surp_RNNG_LC
RNNG_LC_Frontal_Inf_Tri_L = lmer(
  Frontal_Inf_Tri_L ~ RNNG_LC_1_4 + LSTM_seed_1 + ngram_five +  
    sentid + sentpos   + word_freq + word_length + word_rate +　 
    dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Frontal_Inf_Tri_L,
  REML = FALSE
)
summary(RNNG_LC_Frontal_Inf_Tri_L)
coefplot(RNNG_LC_Frontal_Inf_Tri_L)

#LSTM + RNNG_TD + RNNG_LC
LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Tri_L = lmer(
  Frontal_Inf_Tri_L ~ RNNG_LC_1_4 + RNNG_TD_2_10 + LSTM_seed_1  + ngram_five + 
                      sentid + sentpos  + word_freq  + word_length + word_rate + 
                      dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Frontal_Inf_Tri_L,
  REML = FALSE
)
summary(LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Tri_L)
coefplot(LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Tri_L)


#baseline < ngram_five < LSTM < RNNG_TD < RNNG_LC
anova(baseline_Frontal_Inf_Tri_L,
      ngram_five_Frontal_Inf_Tri_L,
      LSTM_Frontal_Inf_Tri_L,
      RNNG_TD_Frontal_Inf_Tri_L,LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Tri_L)
anova(RNNG_LC_Frontal_Inf_Tri_L,LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Tri_L)

#dis_RNNG_LC
dis_RNNG_LC_Frontal_Inf_Tri_L = lmer(
  Frontal_Inf_Tri_L ~ dis_RNNG_LC_1_4 + LSTM_seed_1 + ngram_five  +  
                      sentid + sentpos  + word_freq  + word_length + word_rate+　 
                      dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Frontal_Inf_Tri_L,
  REML = FALSE
)
summary(dis_RNNG_LC_Frontal_Inf_Tri_L)
coefplot(dis_RNNG_LC_Frontal_Inf_Tri_L)
anova(LSTM_Frontal_Inf_Tri_L,dis_RNNG_LC_Frontal_Inf_Tri_L)

#dis_RNNG_TD
dis_RNNG_TD_Frontal_Inf_Tri_L = lmer(
  Frontal_Inf_Tri_L ~ dis_RNNG_TD_2_10 + LSTM_seed_1 + ngram_five  + 
                      sentid + sentpos  + word_freq  + word_length + word_rate+　 
                      dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Frontal_Inf_Tri_L,
  REML = FALSE
)
summary(dis_RNNG_TD_Frontal_Inf_Tri_L)
coefplot(dis_RNNG_TD_Frontal_Inf_Tri_L)
anova(LSTM_Frontal_Inf_Tri_L,dis_RNNG_TD_Frontal_Inf_Tri_L)


#dis_RNNG_TD_LC
dis_RNNG_TD_LC_Frontal_Inf_Tri_L = lmer(
  Frontal_Inf_Tri_L ~ dis_RNNG_LC_1_4 + dis_RNNG_TD_2_10 + LSTM_seed_1 + ngram_five  +  
                      sentid + sentpos  + word_freq  + word_length + word_rate+　 
                      dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Frontal_Inf_Tri_L,
  REML = FALSE
)
summary(dis_RNNG_TD_LC_Frontal_Inf_Tri_L)
coefplot(dis_RNNG_TD_LC_Frontal_Inf_Tri_L)
anova(dis_RNNG_TD_Frontal_Inf_Tri_L,dis_RNNG_TD_LC_Frontal_Inf_Tri_L)#LC>TD
anova(dis_RNNG_LC_Frontal_Inf_Tri_L,dis_RNNG_TD_LC_Frontal_Inf_Tri_L)#LC<TD

########### (3) Frontal_Inf_Orb_L ###################
baseline_Frontal_Inf_Orb_L = lmer(
  Frontal_Inf_Orb_L ~  sentpos + sentid + word_freq + word_length + word_rate + 
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  data,
  REML = FALSE
)
summary(baseline_Frontal_Inf_Orb_L)
coefplot(baseline_Frontal_Inf_Orb_L)

dataTrim_Frontal_Inf_Orb_L = data[scale(resid(baseline_Frontal_Inf_Orb_L)) < 3.0 , ]
nrow(data) - nrow(dataTrim_Frontal_Inf_Orb_L)
(nrow(data) - nrow(dataTrim_Frontal_Inf_Orb_L)) / nrow(data)

baseline_Frontal_Inf_Orb_L = lmer(
  Frontal_Inf_Orb_L ~ sentpos + sentid +  word_freq + word_length +word_rate +
                      dx + dy + dz + rx + ry+ rz  + (1 |subject_number),
  dataTrim_Frontal_Inf_Orb_L,
  REML = FALSE
)
summary(baseline_Frontal_Inf_Orb_L)
coefplot(baseline_Frontal_Inf_Orb_L)


####### models #########
# ngram-five
ngram_five_Frontal_Inf_Orb_L = lmer(
  Frontal_Inf_Orb_L ~  ngram_five + 
                        sentpos + sentid + word_freq + word_length + word_rate +
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Frontal_Inf_Orb_L,
  REML = FALSE
)
summary(ngram_five_Frontal_Inf_Orb_L)
coefplot(ngram_five_Frontal_Inf_Orb_L)
anova(baseline_Frontal_Inf_Orb_L,ngram_five_Frontal_Inf_Orb_L)

# LSTM 
LSTM_Frontal_Inf_Orb_L = lmer(
  Frontal_Inf_Orb_L ~ LSTM_seed_1 + ngram_five + 
                      sentid + sentpos + word_freq + word_length + word_rate +
                      dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Frontal_Inf_Orb_L,
  REML = FALSE
)
summary(LSTM_Frontal_Inf_Orb_L)
coefplot(LSTM_Frontal_Inf_Orb_L)
anova(ngram_five_Frontal_Inf_Orb_L,LSTM_Frontal_Inf_Orb_L)

#surp_RNNG_TD
RNNG_TD_Frontal_Inf_Orb_L = lmer(
  Frontal_Inf_Orb_L ~ RNNG_TD_2_10 + LSTM_seed_1 + ngram_five + 
                      sentid + sentpos   + word_freq + word_length +word_rate +
                      dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Frontal_Inf_Orb_L,
  REML = FALSE
)
summary(RNNG_TD_Frontal_Inf_Orb_L)
coefplot(RNNG_TD_Frontal_Inf_Orb_L)

#surp_RNNG_LC
RNNG_LC_Frontal_Inf_Orb_L = lmer(
  Frontal_Inf_Orb_L ~ RNNG_LC_1_4 + LSTM_seed_1 + ngram_five + 
                      sentid + sentpos + word_freq + word_length +word_rate +
                      dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Frontal_Inf_Orb_L,
  REML = FALSE
)
summary(RNNG_LC_Frontal_Inf_Orb_L)
coefplot(RNNG_LC_Frontal_Inf_Orb_L)

#LSTM + RNNG_TD + RNNG_LC
LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Orb_L = lmer(
  Frontal_Inf_Orb_L ~ RNNG_LC_1_4 + RNNG_TD_2_10 + LSTM_seed_1  + ngram_five + 
                      sentid + sentpos + word_freq + word_length +word_rate + 
                      dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Frontal_Inf_Orb_L,
  REML = FALSE
)
summary(LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Orb_L)
coefplot(LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Orb_L)

#baseline < ngram_five < LSTM < RNNG_TD < RNNG_LC
anova(baseline_Frontal_Inf_Orb_L,
      ngram_five_Frontal_Inf_Orb_L,LSTM_Frontal_Inf_Orb_L,
      RNNG_TD_Frontal_Inf_Orb_L,LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Orb_L)
anova(RNNG_LC_Frontal_Inf_Orb_L,LSTM_RNNG_TD_RNNG_LC_Frontal_Inf_Orb_L)


#dis_RNNG_TD
dis_RNNG_TD_Frontal_Inf_Orb_L = lmer(
  Frontal_Inf_Orb_L ~ dis_RNNG_TD_2_10 + LSTM_seed_1 + ngram_five +  
                      sentid + sentpos + word_freq + word_length +word_rate +　
                      dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Frontal_Inf_Orb_L,
  REML = FALSE
)
summary(dis_RNNG_TD_Frontal_Inf_Orb_L)
coefplot(dis_RNNG_TD_Frontal_Inf_Orb_L)
anova(LSTM_Frontal_Inf_Orb_L,dis_RNNG_TD_Frontal_Inf_Orb_L)

#dis_RNNG_LC
dis_RNNG_LC_Frontal_Inf_Orb_L = lmer(
  Frontal_Inf_Orb_L ~ dis_RNNG_LC_1_4 + LSTM_seed_1 + ngram_five +
                      sentid + sentpos + word_freq + word_length + word_rate +
                      dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Frontal_Inf_Orb_L,
  REML = FALSE
)
summary(dis_RNNG_LC_Frontal_Inf_Orb_L)
coefplot(dis_RNNG_LC_Frontal_Inf_Orb_L)
anova(LSTM_Frontal_Inf_Orb_L,dis_RNNG_LC_Frontal_Inf_Orb_L)


#dis_RNNG_LC + TD 
dis_RNNG_TD_LC_Frontal_Inf_Orb_L = lmer(
  Frontal_Inf_Orb_L ~ dis_RNNG_LC_1_4 + dis_RNNG_TD_2_10 + LSTM_seed_1 + ngram_five + 
                      sentid + sentpos + word_freq + word_length + word_rate +　
                      dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Frontal_Inf_Orb_L,
  REML = FALSE
)
summary(dis_RNNG_LC_Frontal_Inf_Orb_L)
coefplot(dis_RNNG_LC_Frontal_Inf_Orb_L)
anova(dis_RNNG_TD_Frontal_Inf_Orb_L,dis_RNNG_TD_LC_Frontal_Inf_Orb_L) #LC>TD
anova(dis_RNNG_LC_Frontal_Inf_Orb_L,dis_RNNG_TD_LC_Frontal_Inf_Orb_L) #TD<LC


########### (4) Parietal_Inf_L ###################
baseline_Parietal_Inf_L = lmer(
  Parietal_Inf_L ~  sentpos + sentid +  word_freq + word_length +word_rate+
                    dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  data,
  REML = FALSE
)
summary(baseline_Parietal_Inf_L)
coefplot(baseline_Parietal_Inf_L)

dataTrim_Parietal_Inf_L = data[scale(resid(baseline_Parietal_Inf_L)) < 3.0 , ]
nrow(data) - nrow(dataTrim_Parietal_Inf_L)  
(nrow(data) - nrow(dataTrim_Parietal_Inf_L)) / nrow(data)  

baseline_Parietal_Inf_L = lmer(
  Parietal_Inf_L ~  sentpos + sentid + word_freq + word_length + word_rate + 
                    dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Parietal_Inf_L,
  REML = FALSE
)
summary(baseline_Parietal_Inf_L)
coefplot(baseline_Parietal_Inf_L)

####### models #########
# ngram-five
ngram_five_Parietal_Inf_L = lmer(
  Parietal_Inf_L ~  ngram_five + 
                    sentpos + sentid + word_freq + word_length +word_rate + 
                    dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Parietal_Inf_L,
  REML = FALSE
)
summary(ngram_five_Parietal_Inf_L)
coefplot(ngram_five_Parietal_Inf_L)

# LSTM 
LSTM_Parietal_Inf_L = lmer(
  Parietal_Inf_L ~ LSTM_seed_1 + ngram_five + 
                    sentid + sentpos + word_freq + word_length + word_rate +
                    dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Parietal_Inf_L,
  REML = FALSE
)
summary(LSTM_Parietal_Inf_L)
coefplot(LSTM_Parietal_Inf_L)
anova(ngram_five_Parietal_Inf_L,LSTM_Parietal_Inf_L)


#surp_RNNG_TD
RNNG_TD_Parietal_Inf_L = lmer(
  Parietal_Inf_L ~ RNNG_TD_2_10 + LSTM_seed_1 + ngram_five +  
                   sentid + sentpos + word_freq + word_length + word_rate +　 
                    dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Parietal_Inf_L,
  REML = FALSE
)
summary(RNNG_TD_Parietal_Inf_L)
coefplot(RNNG_TD_Parietal_Inf_L)

#surp_RNNG_LC
RNNG_LC_Parietal_Inf_L = lmer(
  Parietal_Inf_L ~ RNNG_LC_1_4 + LSTM_seed_1 + ngram_five +  
                  sentid + sentpos + word_freq + word_length + word_rate +　
                  dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Parietal_Inf_L,
  REML = FALSE
)
summary(RNNG_LC_Parietal_Inf_L)
coefplot(RNNG_LC_Parietal_Inf_L)

#LSTM + RNNG_TD + RNNG_LC
LSTM_RNNG_TD_RNNG_LC_Parietal_Inf_L = lmer(
  Parietal_Inf_L ~ RNNG_LC_1_4 + RNNG_TD_2_10 + LSTM_seed_1  + ngram_five +
                  sentid + sentpos + word_freq + word_length + word_rate + 
                  dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Parietal_Inf_L,
  REML = FALSE
)
summary(LSTM_RNNG_TD_RNNG_LC_Parietal_Inf_L)
coefplot(LSTM_RNNG_TD_RNNG_LC_Parietal_Inf_L)

#baseline < ngram_five < LSTM < RNNG_TD < RNNG_LC
anova(baseline_Parietal_Inf_L,ngram_five_Parietal_Inf_L,
      LSTM_Parietal_Inf_L,RNNG_TD_Parietal_Inf_L,
      LSTM_RNNG_TD_RNNG_LC_Parietal_Inf_L)
anova(RNNG_LC_Parietal_Inf_L,
      LSTM_RNNG_TD_RNNG_LC_Parietal_Inf_L)

# dis_RNNG_TD
dis_RNNG_TD_Parietal_Inf_L = lmer(
  Parietal_Inf_L ~ dis_RNNG_TD_2_10 + LSTM_seed_1 + ngram_five  +  
                    sentid + sentpos + word_freq + word_length + word_rate +　
                    dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Parietal_Inf_L,
  REML = FALSE
)
summary(dis_RNNG_TD_Parietal_Inf_L)
coefplot(dis_RNNG_TD_Parietal_Inf_L)
anova(LSTM_Parietal_Inf_L,dis_RNNG_TD_Parietal_Inf_L)


# dis_RNNG_LC
dis_RNNG_LC_Parietal_Inf_L = lmer(
  Parietal_Inf_L ~ dis_RNNG_LC_1_4 + LSTM_seed_1 + ngram_five  +  
                    sentid + sentpos  + word_freq  + word_length +word_rate +
                    dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Parietal_Inf_L,
  REML = FALSE
)
summary(dis_RNNG_LC_Parietal_Inf_L)
coefplot(dis_RNNG_LC_Parietal_Inf_L)
anova(LSTM_Parietal_Inf_L,dis_RNNG_LC_Parietal_Inf_L)

# dis_RNNG_TD_LC
dis_RNNG_TD_LC_Parietal_Inf_L = lmer(
  Parietal_Inf_L ~ dis_RNNG_LC_1_4 + dis_RNNG_TD_2_10 + LSTM_seed_1 + ngram_five  +
                    sentid + sentpos  + word_freq  + word_length +word_rate +　
                    dx + dy + dz + rx + ry+ rz + (1 |subject_number) ,
  dataTrim_Parietal_Inf_L,
  REML = FALSE
)
summary(dis_RNNG_TD_LC_Parietal_Inf_L)
coefplot(dis_RNNG_TD_LC_Parietal_Inf_L)
anova(dis_RNNG_TD_Parietal_Inf_L,dis_RNNG_TD_LC_Parietal_Inf_L) #TD<LC
anova(dis_RNNG_LC_Parietal_Inf_L,dis_RNNG_TD_LC_Parietal_Inf_L) #LC<TD



########### (5) Angular_L ###################
baseline_Angular_L = lmer(
  Angular_L ~  sentpos + sentid +  word_freq + word_length + word_rate + 
                dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  data,
  REML = FALSE
)
summary(baseline_Angular_L)
coefplot(baseline_Angular_L)

dataTrim_Angular_L = data[scale(resid(baseline_Angular_L)) < 3.0 , ]
nrow(data) - nrow(dataTrim_Angular_L) 
(nrow(data) - nrow(dataTrim_Angular_L)) / nrow(data)  

baseline_Angular_L = lmer(
  Angular_L ~    sentpos + sentid + word_freq + word_length +word_rate + 
                  dx + dy + dz + rx + ry+ rz  + (1 |subject_number),
  dataTrim_Angular_L,
  REML = FALSE
)
summary(baseline_Angular_L)
coefplot(baseline_Angular_L)


####### models #########
# ngram-five
ngram_five_Angular_L = lmer(
  Angular_L ~  ngram_five + 
              sentid + sentpos +  word_freq + word_length + word_rate + 
              dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Angular_L,
  REML = FALSE
)
summary(ngram_five_Angular_L)
coefplot(ngram_five_Angular_L)
anova(baseline_Angular_L,ngram_five_Angular_L)

# LSTM 
LSTM_Angular_L = lmer(
  Angular_L ~ LSTM_seed_1 + ngram_five +  
              sentid + sentpos  + word_freq  + word_length + word_rate +
              dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Angular_L,
  REML = FALSE
)
summary(LSTM_Angular_L)
coefplot(LSTM_Angular_L)
anova(ngram_five_Angular_L,LSTM_Angular_L)


#RNNG_TD
RNNG_TD_Angular_L = lmer(
  Angular_L ~ RNNG_TD_2_10 + LSTM_seed_1 + ngram_five +  
              sentid + sentpos + word_freq + word_length + word_rate +　
              dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Angular_L,
  REML = FALSE
)
summary(RNNG_TD_Angular_L)
coefplot(RNNG_TD_Angular_L)

#RNNG_LC
RNNG_LC_Angular_L = lmer(
  Angular_L ~ RNNG_LC_1_4 + LSTM_seed_1 + ngram_five +  
              sentid + sentpos + word_freq +  word_length + word_rate +　 
              dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Angular_L,
  REML = FALSE
)
summary(RNNG_LC_Angular_L)
coefplot(RNNG_LC_Angular_L)

#LSTM + RNNG_TD + RNNG_LC
LSTM_RNNG_TD_RNNG_LC_Angular_L = lmer(
  Angular_L ~ RNNG_LC_1_4 + RNNG_TD_2_10 + LSTM_seed_1  + ngram_five + 
              sentid + sentpos  + word_freq + word_length + word_rate + 
              dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Angular_L,
  REML = FALSE
)
summary(LSTM_RNNG_TD_RNNG_LC_Angular_L)
coefplot(LSTM_RNNG_TD_RNNG_LC_Angular_L)

#baseline < ngram_five < LSTM < RNNG_TD < RNNG_LC
anova(baseline_Angular_L,ngram_five_Angular_L,
      LSTM_Angular_L,RNNG_TD_Angular_L,
      LSTM_RNNG_TD_RNNG_LC_Angular_L)
anova(RNNG_LC_Angular_L,
      LSTM_RNNG_TD_RNNG_LC_Angular_L)

# dis_RNNG_TD
dis_RNNG_TD_Angular_L = lmer(
  Angular_L ~ dis_RNNG_TD_2_10 + LSTM_seed_1 + ngram_five + 
              sentid + sentpos  + word_freq + word_length + word_rate +　
              dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Angular_L,
  REML = FALSE
)
summary(dis_RNNG_TD_Angular_L)
coefplot(dis_RNNG_TD_Angular_L)
anova(LSTM_Angular_L,dis_RNNG_TD_Angular_L)

# dis_RNNG_LC
dis_RNNG_LC_Angular_L = lmer(
  Angular_L ~ dis_RNNG_LC_1_4 + LSTM_seed_1 + ngram_five +  
              sentid + sentpos  + word_freq + word_length + word_rate +
              dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Angular_L,
  REML = FALSE
)
summary(dis_RNNG_LC_Angular_L)
coefplot(dis_RNNG_LC_Angular_L)
anova(LSTM_Angular_L,dis_RNNG_LC_Angular_L)

# dis_RNNG_LC_TD
dis_RNNG_TD_LC_Angular_L = lmer(
  Angular_L ~ dis_RNNG_LC_1_4 + dis_RNNG_TD_2_10 + LSTM_seed_1 + ngram_five +  
              sentid + sentpos  + word_freq  + word_length + word_rate +　
              dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Angular_L,
  REML = FALSE
)
summary(dis_RNNG_TD_LC_Angular_L)
coefplot(dis_RNNG_TD_LC_Angular_L)
anova(dis_RNNG_TD_Angular_L,dis_RNNG_TD_LC_Angular_L)
anova(dis_RNNG_LC_Angular_L,dis_RNNG_TD_LC_Angular_L)


########### (6) Temporal_Sup_L ###################
baseline_Temporal_Sup_L = lmer(
  Temporal_Sup_L ~  sentpos + sentid + word_freq + word_length + word_rate + 
                    dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  data,
  REML = FALSE
)
summary(baseline_Temporal_Sup_L)
coefplot(baseline_Temporal_Sup_L)

dataTrim_Temporal_Sup_L = data[scale(resid(baseline_Temporal_Sup_L)) < 3.0 , ]
nrow(data) - nrow(dataTrim_Temporal_Sup_L)  
(nrow(data) - nrow(dataTrim_Temporal_Sup_L)) / nrow(data) 

 
baseline_Temporal_Sup_L = lmer(
  Temporal_Sup_L ~ sentpos + sentid +  word_freq + word_length + word_rate + 
                    dx + dy + dz + rx + ry+ rz  + (1 |subject_number),
  dataTrim_Temporal_Sup_L,
  REML = FALSE
)
summary(baseline_Temporal_Sup_L)
coefplot(baseline_Temporal_Sup_L)


####### models #########
# ngram-five
ngram_five_Temporal_Sup_L = lmer(
  Temporal_Sup_L ~  ngram_five + 
                    sentpos + sentid +  word_freq  + word_length + word_rate + 
                    dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Temporal_Sup_L,
  REML = FALSE
)
summary(ngram_five_Temporal_Sup_L)
coefplot(ngram_five_Temporal_Sup_L)
anova(baseline_Temporal_Sup_L,ngram_five_Temporal_Sup_L)

# LSTM 
LSTM_Temporal_Sup_L = lmer(
  Temporal_Sup_L ~ LSTM_seed_1 + ngram_five +  
                    sentid + sentpos  + word_freq + word_length + word_rate +　
                    dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Temporal_Sup_L,
  REML = FALSE
)
summary(LSTM_Temporal_Sup_L)
coefplot(LSTM_Temporal_Sup_L)
anova(ngram_five_Temporal_Sup_L,LSTM_Temporal_Sup_L)

#surp_RNNG_TD
RNNG_TD_Temporal_Sup_L = lmer(
  Temporal_Sup_L ~ RNNG_TD_2_10 + LSTM_seed_1 + ngram_five +  
    sentid + sentpos + word_freq + word_length +  word_rate +　 
    dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Temporal_Sup_L,
  REML = FALSE
)
summary(RNNG_TD_Temporal_Sup_L)
coefplot(RNNG_TD_Temporal_Sup_L)

#surp_RNNG_LC
RNNG_LC_Temporal_Sup_L = lmer(
  Temporal_Sup_L ~ RNNG_LC_1_4 + LSTM_seed_1 + ngram_five +  
    sentid + sentpos   + word_freq + word_length + word_rate +　
    dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Temporal_Sup_L,
  REML = FALSE
)
summary(RNNG_LC_Temporal_Sup_L)
coefplot(RNNG_LC_Temporal_Sup_L)

#LSTM + RNNG_TD + RNNG_LC
LSTM_RNNG_TD_RNNG_LC_Temporal_Sup_L = lmer(
  Temporal_Sup_L ~ RNNG_LC_1_4 + RNNG_TD_2_10 + LSTM_seed_1 + ngram_five + 
    sentid + sentpos  + word_freq  + word_length + word_rate + 
    dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Temporal_Sup_L,
  REML = FALSE
)
summary(LSTM_RNNG_TD_RNNG_LC_Temporal_Sup_L)
coefplot(LSTM_RNNG_TD_RNNG_LC_Temporal_Sup_L)

#baseline < ngram_five < LSTM < RNNG_TD < RNNG_LC
anova(baseline_Temporal_Sup_L,ngram_five_Temporal_Sup_L,
      LSTM_Temporal_Sup_L,RNNG_TD_Temporal_Sup_L,
      LSTM_RNNG_TD_RNNG_LC_Temporal_Sup_L)
anova(RNNG_LC_Temporal_Sup_L,
      LSTM_RNNG_TD_RNNG_LC_Temporal_Sup_L)

# dis_RNNG_TD
dis_RNNG_TD_Temporal_Sup_L = lmer(
  Temporal_Sup_L ~ dis_RNNG_TD_2_10 + LSTM_seed_1 + ngram_five +  
                    sentid + sentpos  + word_freq  + word_length + word_rate +　
                    dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Temporal_Sup_L,
  REML = FALSE
)
summary(dis_RNNG_TD_Temporal_Sup_L)
coefplot(dis_RNNG_TD_Temporal_Sup_L)
anova(LSTM_Temporal_Sup_L,dis_RNNG_TD_Temporal_Sup_L)

# dis_RNNG_LC
dis_RNNG_LC_Temporal_Sup_L = lmer(
  Temporal_Sup_L ~ dis_RNNG_LC_1_4 + LSTM_seed_1 + ngram_five  +
                    sentid + sentpos  + word_freq  + word_length + word_rate +　
                    dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Temporal_Sup_L,
  REML = FALSE
)
summary(dis_RNNG_LC_Temporal_Sup_L)
coefplot(dis_RNNG_LC_Temporal_Sup_L)
anova(LSTM_Temporal_Sup_L,dis_RNNG_LC_Temporal_Sup_L)

# dis_RNNG_TD_LC
dis_RNNG_TD_LC_Temporal_Sup_L = lmer(
  Temporal_Sup_L ~ dis_RNNG_LC_1_4 + dis_RNNG_TD_2_10 + LSTM_seed_1 + ngram_five  + 
                  sentid + sentpos  + word_freq  + word_length + word_rate +
                  dx + dy + dz + rx + ry+ rz + (1 |subject_number) ,
  dataTrim_Temporal_Sup_L,
  REML = FALSE
)
summary(dis_RNNG_TD_LC_Temporal_Sup_L)
coefplot(dis_RNNG_TD_LC_Temporal_Sup_L)
anova(dis_RNNG_TD_Temporal_Sup_L,dis_RNNG_TD_LC_Temporal_Sup_L) #TD<LC
anova(dis_RNNG_LC_Temporal_Sup_L,dis_RNNG_TD_LC_Temporal_Sup_L) #LC<TD


########### (7) Temporal_Pole_Sup_L ###################
baseline_Temporal_Pole_Sup_L = lmer(
  Temporal_Pole_Sup_L ~  sentpos + sentid + word_freq + word_length + word_rate  +
                          dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  data,
  REML = FALSE
)
summary(baseline_Temporal_Pole_Sup_L)
coefplot(baseline_Temporal_Pole_Sup_L)
 
dataTrim_Temporal_Pole_Sup_L = data[scale(resid(baseline_Temporal_Pole_Sup_L)) < 3.0 , ]
nrow(data) - nrow(dataTrim_Temporal_Pole_Sup_L)  
(nrow(data) - nrow(dataTrim_Temporal_Pole_Sup_L)) / nrow(data) 

 
baseline_Temporal_Pole_Sup_L = lmer(
  Temporal_Pole_Sup_L ~ sentpos + sentid +  word_freq + word_length + word_rate  + 
                        dx + dy + dz + rx + ry+ rz  + (1 |subject_number),
  dataTrim_Temporal_Pole_Sup_L,
  REML = FALSE
)
summary(baseline_Temporal_Pole_Sup_L)
coefplot(baseline_Temporal_Pole_Sup_L)


####### models #########
# ngram-five
ngram_five_Temporal_Pole_Sup_L = lmer(
  Temporal_Pole_Sup_L ~  ngram_five + 
                        sentpos + sentid + word_freq + word_length + word_rate + 
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Temporal_Pole_Sup_L,
  REML = FALSE
)
summary(ngram_five_Temporal_Pole_Sup_L)
coefplot(ngram_five_Temporal_Pole_Sup_L)
anova(baseline_Temporal_Pole_Sup_L,ngram_five_Temporal_Pole_Sup_L)

# LSTM 
LSTM_Temporal_Pole_Sup_L = lmer(
  Temporal_Pole_Sup_L ~ LSTM_seed_1 + ngram_five +  
                        sentid + sentpos + word_freq + word_length + word_rate +　
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number) ,
  dataTrim_Temporal_Pole_Sup_L,
  REML = FALSE
)
summary(LSTM_Temporal_Pole_Sup_L)
coefplot(LSTM_Temporal_Pole_Sup_L)
anova(ngram_five_Temporal_Pole_Sup_L,LSTM_Temporal_Pole_Sup_L)

#surp_RNNG_TD
RNNG_TD_Temporal_Pole_Sup_L = lmer(
  Temporal_Pole_Sup_L ~ RNNG_TD_2_10 + LSTM_seed_1 + ngram_five + 
                        sentid + sentpos + word_freq + word_length + word_rate +　
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Temporal_Pole_Sup_L,
  REML = FALSE
)
summary(RNNG_TD_Temporal_Pole_Sup_L)
coefplot(RNNG_TD_Temporal_Pole_Sup_L)

#surp_RNNG_LC
RNNG_LC_Temporal_Pole_Sup_L = lmer(
  Temporal_Pole_Sup_L ~ RNNG_LC_1_4 + LSTM_seed_1 + ngram_five +  
                        sentid + sentpos + word_freq + word_length + word_rate +　 
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Temporal_Pole_Sup_L,
  REML = FALSE
)
summary(RNNG_LC_Temporal_Pole_Sup_L)
coefplot(RNNG_LC_Temporal_Pole_Sup_L)

#LSTM + RNNG_TD + RNNG_LC
LSTM_RNNG_TD_RNNG_LC_Temporal_Pole_Sup_L = lmer(
  Temporal_Pole_Sup_L ~ RNNG_LC_1_4 + RNNG_TD_2_10 + LSTM_seed_1  + ngram_five + 
                        sentid + sentpos  + word_freq  + word_length + word_rate + 
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Temporal_Pole_Sup_L,
  REML = FALSE
)
summary(LSTM_RNNG_TD_RNNG_LC_Temporal_Pole_Sup_L)
coefplot(LSTM_RNNG_TD_RNNG_LC_Temporal_Pole_Sup_L)

#baseline < ngram_five < LSTM < RNNG_TD < RNNG_LC
anova(baseline_Temporal_Pole_Sup_L,ngram_five_Temporal_Pole_Sup_L,
      LSTM_Temporal_Pole_Sup_L,RNNG_TD_Temporal_Pole_Sup_L,
      LSTM_RNNG_TD_RNNG_LC_Temporal_Pole_Sup_L)
anova(RNNG_LC_Temporal_Pole_Sup_L,
      LSTM_RNNG_TD_RNNG_LC_Temporal_Pole_Sup_L)#surp_RNNG_LC < surp_RNNG_TD

#dis_RNNG_TD
dis_RNNG_TD_Temporal_Pole_Sup_L = lmer(
  Temporal_Pole_Sup_L ~ dis_RNNG_TD_2_10 + LSTM_seed_1 + ngram_five  +  
                        sentid + sentpos  + word_freq  + word_length + word_rate +　 
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Temporal_Pole_Sup_L,
  REML = FALSE
)
summary(dis_RNNG_TD_Temporal_Pole_Sup_L)
coefplot(dis_RNNG_TD_Temporal_Pole_Sup_L)
anova(LSTM_Temporal_Pole_Sup_L,dis_RNNG_TD_Temporal_Pole_Sup_L)


#dis_RNNG_LC
dis_RNNG_LC_Temporal_Pole_Sup_L = lmer(
  Temporal_Pole_Sup_L ~ dis_RNNG_LC_1_4 + LSTM_seed_1 + ngram_five  +  
                        sentid + sentpos + word_freq + word_length + word_rate +　 
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Temporal_Pole_Sup_L,
  REML = FALSE
)
summary(dis_RNNG_LC_Temporal_Pole_Sup_L)
coefplot(dis_RNNG_LC_Temporal_Pole_Sup_L)
anova(LSTM_Temporal_Pole_Sup_L,dis_RNNG_LC_Temporal_Pole_Sup_L)

#dis_RNNG_TD_LC
dis_RNNG_TD_LC_Temporal_Pole_Sup_L = lmer(
  Temporal_Pole_Sup_L ~ dis_RNNG_LC_1_4 + dis_RNNG_TD_2_10 + LSTM_seed_1 + ngram_five  +  
                        sentid + sentpos  + word_freq  + word_length + word_rate +　 
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Temporal_Pole_Sup_L,
  REML = FALSE
)
summary(dis_RNNG_TD_LC_Temporal_Pole_Sup_L)
coefplot(dis_RNNG_TD_LC_Temporal_Pole_Sup_L)
anova(dis_RNNG_TD_Temporal_Pole_Sup_L,dis_RNNG_TD_LC_Temporal_Pole_Sup_L)
anova(dis_RNNG_LC_Temporal_Pole_Sup_L,dis_RNNG_TD_LC_Temporal_Pole_Sup_L)


########### (8) Temporal_Pole_Mid_L ###################
baseline_Temporal_Pole_Mid_L = lmer(
  Temporal_Pole_Mid_L ~  sentpos + sentid + word_freq + word_length + word_rate +
                          dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  data,
  REML = FALSE
)
summary(baseline_Temporal_Pole_Mid_L)
coefplot(baseline_Temporal_Pole_Mid_L)

dataTrim_Temporal_Pole_Mid_L = data[scale(resid(baseline_Temporal_Pole_Mid_L)) < 3.0 , ]
nrow(data) - nrow(dataTrim_Temporal_Pole_Mid_L)  
(nrow(data) - nrow(dataTrim_Temporal_Pole_Mid_L)) / nrow(data)  

baseline_Temporal_Pole_Mid_L = lmer(
  Temporal_Pole_Mid_L ~ sentpos + sentid + word_freq + word_length + word_rate + 
                        dx + dy + dz + rx + ry+ rz  + (1 |subject_number),
  dataTrim_Temporal_Pole_Mid_L,
  REML = FALSE
)
summary(baseline_Temporal_Pole_Mid_L)
coefplot(baseline_Temporal_Pole_Mid_L)


####### models #########
# ngram-five
ngram_five_Temporal_Pole_Mid_L = lmer(
  Temporal_Pole_Mid_L ~  ngram_five + 
                        sentpos + sentid + word_freq + word_length + word_rate + 
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Temporal_Pole_Mid_L,
  REML = FALSE
)
summary(ngram_five_Temporal_Pole_Mid_L)
coefplot(ngram_five_Temporal_Pole_Mid_L)
anova(baseline_Temporal_Pole_Mid_L,ngram_five_Temporal_Pole_Mid_L)

# LSTM 
LSTM_Temporal_Pole_Mid_L = lmer(
  Temporal_Pole_Mid_L ~ LSTM_seed_1 + ngram_five  +  
                        sentid + sentpos + word_freq + word_length + word_rate +　 
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Temporal_Pole_Mid_L,
  REML = FALSE
)
summary(LSTM_Temporal_Pole_Mid_L)
coefplot(LSTM_Temporal_Pole_Mid_L)
anova(ngram_five_Temporal_Pole_Mid_L,LSTM_Temporal_Pole_Mid_L)


#surp_RNNG_TD
RNNG_TD_Temporal_Pole_Mid_L = lmer(
  Temporal_Pole_Mid_L ~ RNNG_TD_2_10 + LSTM_seed_1 + ngram_five +  
                        sentid + sentpos + word_freq + word_length + word_rate +　 
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Temporal_Pole_Mid_L,
  REML = FALSE
)
summary(RNNG_TD_Temporal_Pole_Mid_L)
coefplot(RNNG_TD_Temporal_Pole_Mid_L)

#surp_RNNG_LC
RNNG_LC_Temporal_Pole_Mid_L = lmer(
  Temporal_Pole_Mid_L ~ RNNG_LC_1_4 + LSTM_seed_1 + ngram_five + 
                        sentid + sentpos + word_freq +  word_length + word_rate +　 
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Temporal_Pole_Mid_L,
  REML = FALSE
)
summary(RNNG_LC_Temporal_Pole_Mid_L)
coefplot(RNNG_LC_Temporal_Pole_Mid_L)

#LSTM + RNNG_TD + RNNG_LC
LSTM_RNNG_TD_RNNG_LC_Temporal_Pole_Mid_L = lmer(
  Temporal_Pole_Mid_L ~ RNNG_LC_1_4 + RNNG_TD_2_10 + LSTM_seed_1 + ngram_five + 
                        sentid + sentpos + word_freq + word_length+ word_rate + 
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Temporal_Pole_Mid_L,
  REML = FALSE
)
summary(LSTM_RNNG_TD_RNNG_LC_Temporal_Pole_Mid_L)
coefplot(LSTM_RNNG_TD_RNNG_LC_Temporal_Pole_Mid_L)

#baseline < ngram_five < LSTM < RNNG_TD < RNNG_LC
anova(baseline_Temporal_Pole_Mid_L,ngram_five_Temporal_Pole_Mid_L,
      LSTM_Temporal_Pole_Mid_L,RNNG_TD_Temporal_Pole_Mid_L,
      LSTM_RNNG_TD_RNNG_LC_Temporal_Pole_Mid_L)
anova(RNNG_LC_Temporal_Pole_Mid_L,
      LSTM_RNNG_TD_RNNG_LC_Temporal_Pole_Mid_L) # surp_RNNG_LC < surp_RNNG_TD

# dis_RNNG_TD
dis_RNNG_TD_Temporal_Pole_Mid_L = lmer(
  Temporal_Pole_Mid_L ~ dis_RNNG_TD_2_10 + LSTM_seed_1 + ngram_five +  
                        sentid + sentpos + word_freq + word_length + word_rate +　 
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Temporal_Pole_Mid_L,
  REML = FALSE
)
summary(dis_RNNG_TD_Temporal_Pole_Mid_L)
coefplot(dis_RNNG_TD_Temporal_Pole_Mid_L)
anova(LSTM_Temporal_Pole_Mid_L,dis_RNNG_TD_Temporal_Pole_Mid_L)

# dis_RNNG_LC
dis_RNNG_LC_Temporal_Pole_Mid_L = lmer(
  Temporal_Pole_Mid_L ~ dis_RNNG_LC_1_4 + LSTM_seed_1 + ngram_five  +  
                        sentid + sentpos + word_freq + word_length + word_rate +　 
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number),
  dataTrim_Temporal_Pole_Mid_L,
  REML = FALSE
)
summary(dis_RNNG_LC_Temporal_Pole_Mid_L)
coefplot(dis_RNNG_LC_Temporal_Pole_Mid_L)
anova(LSTM_Temporal_Pole_Mid_L,dis_RNNG_LC_Temporal_Pole_Mid_L)

# dis_RNNG_TD_LC
dis_RNNG_TD_LC_Temporal_Pole_Mid_L = lmer(
  Temporal_Pole_Mid_L ~ dis_RNNG_LC_1_4 + dis_RNNG_TD_2_10 + LSTM_seed_1 + ngram_five  +  
                        sentid + sentpos + word_freq  + word_length + word_rate +　 
                        dx + dy + dz + rx + ry+ rz + (1 |subject_number) ,
  dataTrim_Temporal_Pole_Mid_L,
  REML = FALSE
)
summary(dis_RNNG_TD_LC_Temporal_Pole_Mid_L)
coefplot(dis_RNNG_TD_LC_Temporal_Pole_Mid_L)
anova(dis_RNNG_LC_Temporal_Pole_Mid_L,dis_RNNG_TD_LC_Temporal_Pole_Mid_L)
anova(dis_RNNG_TD_Temporal_Pole_Mid_L,dis_RNNG_TD_LC_Temporal_Pole_Mid_L)


#----------------------#
#Bonferroi correction
# Threshold = a/N where a = e.g., 0.05, N = the number of tests 
# e.g., 0.05/8 (the number of ROIs) = 0.00625