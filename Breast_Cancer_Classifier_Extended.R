library(keras)
library(tidyverse)
library(recipes)
library(ROCR)
library(mlbench)
library(DataExplorer)
library(tidyverse)
library(polycor)
library(car)
library(broom)
library(dplyr)
library(rsample)
library(class)

raw_data<-wdbc%>%select(-Id)

dim(raw_data)

glimpse(raw_data)

plot_missing(raw_data)
hetcor(raw_data)



set.seed(1998)

train_test_split<-initial_split(raw_data, prop=0.5)

train_tbl<-training(train_test_split)
test_tbl<-testing(train_test_split)

train_tbl<-train_tbl[-c(155),]

rec_obj<-recipe(Class ~., data=train_tbl)%>%
  prep(data=train_tbl)

train_ready<-bake(rec_obj, new_data=train_tbl)
test_ready<-bake(rec_obj, new_data=test_tbl)

################MODEL 1################
log.fit<-glm(Class ~ ., data=train_ready, family=binomial(link="logit"),maxit=100)
summary(log.fit)
#######################################

################MODEL 2################
log.fit<-glm(Class ~ .
             -CompactW    
             -ConcavityW   
             -Concave.pointsW
             -SymmetryW   
             -FractalW     
             -PerimeterW
             -Concave.pointsSE
             -SymmetrySE  
             -FractalSE
             -CompactSE
             -PerimeterSE    
             -AreaSE 
             -Symmetry     
             -Fractal   
             -RadiusSE  
             -Compact
             -SmoothW
             -AreaW
             -TextureW
             -Radius
             -Area
             -Perimeter
             , data=train_ready, family=binomial(link="logit"),maxit=100)
summary(log.fit)
########################################
anova(log.fit,test="Chisq")
vif(log.fit) #delete predictors larger than 10
durbinWatsonTest(log.fit)#if p value > 0.05 we fail to reject the null hyp which means we are good

###############################ACCURACY###########################
log.prob <-predict(log.fit, newdata=test_ready, type='response')
log.pred <-ifelse(log.prob >0.5, "M","B")


table(log.pred, test_ready$Class)
mean(log.pred==test_ready$Class)
##################################################################

###########DIAGNOSTICS#################
outlierTest(log.fit)
#True and False Neg and Pos value
p <- predict(log.fit, newdata=test_ready, type="response")
pr <- prediction(p, test_ready$Class)
prf <- performance(pr, measure = "tnr", x.measure = "fnr")
plot(prf)



auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc


### Logistic Diagnostics
# Linearity

probabilities <- predict(log.fit, newdata=train_ready, type='response')
predict.classes <- ifelse(probabilities > 0.5, "M", "B")

mydata <- train_ready %>%
  dplyr::select_if(is.numeric) 
predictors <- colnames(mydata)

# Bind the logit and tidying the data for plot
mydata <- mydata %>%
  mutate(logit = log(probabilities/(1-probabilities))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)

ggplot(mydata, aes(logit, predictor.value))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  theme_bw() + 
  facet_wrap(~predictors, scales = "free_y")


# Outliers and Influence Points
plot(log.fit, which = 4, id.n = 3)
# Extract model results
model.data <- augment(log.fit) %>% 
  mutate(index = 1:n()) 

model.data %>% top_n(3, .cooksd)

outlierTest(log.fit)



########################################
########################KNN#############################
train.X<-cbind(train_ready%>%select(everything()))
test.X<-cbind(test_ready%>%select(everything()))

train.label<-as.vector(ifelse(pull(train_ready, Class) == "M",1,0))
train.X<-train.X%>%select(-Class)
test.X<-test.X%>%select(-Class)
train.label
set.seed(1)

knn.pred<-knn(train.X,test.X,train.label,k=50)

table(knn.pred,test_ready$Class)

################IMPORTANT
####PLEASE NOTE!!!!!!!!!!!!!!!!!!<<<<<<<<
#lda requires MASS, but MASS has a select
#Function that conflicts with select used in knn
#Please remove mass before running KNN 
#That is why lda model is after KNN
###############################
#IMPORTANT^^^^^^^^^^^^


library(MASS)
############LDA#############

lda.fit<-lda(formula=Class~.-CompactW    
             -ConcavityW   
             -Concave.pointsW
             -SymmetryW   
             -FractalW     
             -PerimeterW
             -Concave.pointsSE
             -SymmetrySE  
             -FractalSE
             -CompactSE
             -PerimeterSE    
             -AreaSE 
             -Symmetry     
             -Fractal   
             -RadiusSE  
             -Compact
             -SmoothW
             -AreaW
             -TextureW
             -Radius
             -Area
             -Perimeter
             , data=train_ready)
summary(lda.fit)
names(lda.fit)
plot(lda.fit)
lda.pred <-predict(lda.fit,test_ready)
names(lda.pred)
lda.class<-lda.pred$class
##########CONFUSION MATRIX LDA####
table(lda.class, test_ready$Class)
mean(lda.class== test_ready$Class)
