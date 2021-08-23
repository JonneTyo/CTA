require(survival)
require(survminer)


make_pdf_file=T

pdf("KM_plots.pdf")

# Path
setwd("E:/Other/Repos/CTA/Main/CTA-Projekti")


###
# Survival data, all
tmp_data=read.csv("kaplan-meier data.csv")
surv_time=tmp_data$passed.time/365
true_values=tmp_data$true.y.unrestricted.years
surv_test=Surv(time=surv_time,event=true_values)


###
# PCA2
predictions=tmp_data$PCA2
predictions=ifelse(predictions==1,"Value 1",ifelse(predictions==2,"Value 2","Value 3"))

my_data=data.frame(cbind(surv_test,predictions))

fit_lasso=survfit(surv_test~predictions,data=my_data)
ggsurvplot(fit_lasso,data=my_data,pval=F,risk.table=F,conf.int=T,break.time.by=1,censor=F,cumevents=F,title="PCA2", legend.labs=c("Low PCA2 score", "Normal PCA2 score", "High PCA2 score"), palette=c("blue", "orange", "green"))


###
# All
tmp_data=read.csv("predictions all .csv")
predictions=tmp_data$pred.y.unrestricted.years
predictions=ifelse(predictions==0,"Low risk","High risk")

table(predictions)
# High risk  Low risk 
# 929      1355 

my_data=data.frame(cbind(surv_test,predictions))

fit_lasso=survfit(surv_test~predictions,data=my_data)
ggsurvplot(fit_lasso,data=my_data,pval=F,risk.table=F,conf.int=T,break.time.by=1,censor=F,cumevents=F,legend.labs=c("High risk","Low risk"),title="Predictions, all")


###
# All CTA
tmp_data=read.csv("predictions all cta.csv")
predictions=tmp_data$pred.y.unrestricted.years
predictions=ifelse(predictions==0,"Low risk","High risk")

table(predictions)
# High risk  Low risk 
# 891      1393 

my_data=data.frame(cbind(surv_test,predictions))

fit_lasso=survfit(surv_test~predictions,data=my_data)
ggsurvplot(fit_lasso,data=my_data,pval=F,risk.table=F,conf.int=T,break.time.by=1,censor=F,cumevents=F,legend.labs=c("High risk","Low risk"),title="Predictions, all CTA")


###
# Survival data, PET
tmp_data=read.csv("predictions only pet .csv")
inds=tmp_data$X
tmp_data=read.csv("kaplan-meier data.csv")
surv_time=tmp_data$passed[which(tmp_data$X %in% inds)]/365
true_values=tmp_data$true.y.unrestricted.years[which(tmp_data$X %in% inds)]
surv_test=Surv(time=surv_time,event=true_values)


###
# Only PET
tmp_data=read.csv("predictions only pet .csv")
predictions=tmp_data$pred.y.unrestricted.years
predictions=ifelse(predictions==0,"Low risk","High risk")

table(predictions)
# High risk  Low risk 
# 361       452 

my_data=data.frame(cbind(surv_test,predictions))

fit_lasso=survfit(surv_test~predictions,data=my_data)
ggsurvplot(fit_lasso,data=my_data,pval=F,risk.table=F,conf.int=T,break.time.by=1,censor=F,cumevents=F,legend.labs=c("High risk","Low risk"),title="Predictions, only PET")


###
# Only PET CTA
tmp_data=read.csv("predictions only pet cta.csv")
predictions=tmp_data$pred.y.unrestricted.years
predictions=ifelse(predictions==0,"Low risk","High risk")

table(predictions)
# High risk  Low risk 
# 288       525 

my_data=data.frame(cbind(surv_test,predictions))

fit_lasso=survfit(surv_test~predictions,data=my_data)
ggsurvplot(fit_lasso,data=my_data,pval=F,risk.table=F,conf.int=T,break.time.by=1,censor=F,cumevents=F,legend.labs=c("High risk","Low risk"),title="Predictions, only PET CTA")


dev.off()


