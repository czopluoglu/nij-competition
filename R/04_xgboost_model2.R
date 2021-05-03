require(xgboost)
require(caret)
require(pROC)
require(xgboostExplainer)


################################################################################

#https://arxiv.org/pdf/1802.09596.pdf

# Table 3

#nrounds              4168,    920.7,    4550.95
#eta                 0.018,    0.002,      0.355
#subsample           0.839,    0.545,      0.958
#max_depth              13,      5.6,       14
#min_child_weight     2.06,    1.295,      6.984
#colsample_bytree     0.752,   0.419       0.864
#colsample_bylevel    0.585     0.335      0.886
#lambda              0.982      0.008      29.755
#alpha               1.113      0.002      6.105


###############################################################################


final_test <- test_submit[,c(1,34,35,43,69:77,82:92,94,96,98,100,102,103,104,
                             106,108,109,111,113,115:126)]

train_ <- full_data[,colnames(full_data)%in%colnames(final_test)]

out <- full_data$y1

###############################################################################

eval <- function(x,type){
  
  # x, 4 column matrix (gender, race, probability, model prediction, outcome)
  # type, brier.m, brier.f, brier.a, fair.m, fair.f
  # brier.m --> brier score for males
  # brier.f --> brier score for females
  # brier.a --> brier score for all
  # fair.m --> fairness score for males
  # fair.f --> fairness score for females
  
  if(type == 'brier.m'){
    x1  <- x[which(x$gender==0),]
    ret <- mean((x1$pred - x1$out)^2)
  }
  
  if(type == 'brier.f'){
    x2  <- x[which(x$gender==1),]
    ret <- mean((x2$pred - x2$out)^2)
  }
  
  if(type == 'brier.a'){
    ret <- mean((x$pred - x$out)^2)
  }
  
  if(type == 'fair.m'){
    
    x1  <- x[which(x$gender==0),]
    
    ret1 <- mean((x1$pred - x1$out)^2)
    
    x10 <- x1[which(x1$race==0),]
    
    conf.w1 <- matrix(nrow=2,ncol=2)
    conf.w1[1,1] = length(which(x10$out==0 & x10$pred2==0))
    conf.w1[1,2] = length(which(x10$out==0 & x10$pred2==1))
    conf.w1[2,1] = length(which(x10$out==1 & x10$pred2==0))
    conf.w1[2,2] = length(which(x10$out==1 & x10$pred2==1))
    fpw1    <- conf.w1[1,2]/sum(conf.w1[1,]) # false-positive rate for white
    
    x11 <- x1[which(x1$race==1),]
    
    conf.b1 <- matrix(nrow=2,ncol=2)
    conf.b1[1,1] = length(which(x11$out==0 & x11$pred2==0))
    conf.b1[1,2] = length(which(x11$out==0 & x11$pred2==1))
    conf.b1[2,1] = length(which(x11$out==1 & x11$pred2==0))
    conf.b1[2,2] = length(which(x11$out==1 & x11$pred2==1))
    fpb1    <- conf.b1[1,2]/sum(conf.b1[1,]) # false-positive rate for black
    
    ret <- (1-ret1)*(1- (fpb1 - fpw1))
    
  }
  
  if(type == 'fair.f'){
    
    x2  <- x[which(x$gender==1),]
    
    ret1 <- mean((x2$pred - x2$out)^2)
    
    x20 <- x2[which(x2$race==0),]
    
    conf.w1 <- matrix(nrow=2,ncol=2)
    conf.w1[1,1] = length(which(x20$out==0 & x20$pred2==0))
    conf.w1[1,2] = length(which(x20$out==0 & x20$pred2==1))
    conf.w1[2,1] = length(which(x20$out==1 & x20$pred2==0))
    conf.w1[2,2] = length(which(x20$out==1 & x20$pred2==1))
    fpw1    <- conf.w1[1,2]/sum(conf.w1[1,]) # false-positive rate for white
    
    x21 <- x2[which(x2$race==1),]
    
    conf.b1 <- matrix(nrow=2,ncol=2)
    conf.b1[1,1] = length(which(x21$out==0 & x21$pred2==0))
    conf.b1[1,2] = length(which(x21$out==0 & x21$pred2==1))
    conf.b1[2,1] = length(which(x21$out==1 & x21$pred2==0))
    conf.b1[2,2] = length(which(x21$out==1 & x21$pred2==1))
    fpb1    <- conf.b1[1,2]/sum(conf.b1[1,]) # false-positive rate for black
    
    ret <- (1-ret1)*(1- (fpb1 - fpw1))
    
  }
  
  ret
  
}

# Custom objective function

brier <- function(preds, dtrain){
  
  labels <- getinfo(dtrain, "label")
  preds  <- 1 / (1 + exp(-preds))
  
  grad = 2*(preds-labels)*preds*(1-preds)
  hess = 2*(2*(labels+1)*preds-labels-3*preds*preds)*preds*(1-preds)
  
  return(list(grad = grad, hess = hess))
}

# Custom Eval Metrix

brier.err <- function(preds, dtrain){
  labels <- getinfo(dtrain, "label")
  err <- as.numeric(sum(labels != (preds > 0))) / length(labels)
  return(list(metric = "briererr", value = err))
}



###############################################################################

set.seed(05012021)

vec <- 1:nrow(train_)
loc1 <- sample(vec,15000)
loc2 <- vec[!vec%in%loc1]


test  <- train_[loc2,]
train <- train_[loc1,]

out2 <- out[loc2]
out1 <- out[loc1]

################################################################################

dtrain <- xgb.DMatrix(data = data.matrix(train[,-1]), label=out1)
dtest  <- xgb.DMatrix(data = data.matrix(test[,-1]),  label=out2)

################################################################################

myfolds <- createFolds(1:nrow(dtrain),10)

################################################################################


################################################################################
################################################################################
#Stage 1:  Tune eta
################################################################################
################################################################################

params <- list(booster           = "gbtree", 
               gamma             = 0, 
               max_depth         = 6, 
               min_child_weight  = 1, 
               subsample         = 1, 
               colsample_bytree  = 1,
               max_delta_step    = 0,
               lambda            = 1,
               alpha             = 0,
               scale_pos_weight  = 1,
               num_parallel_tree = 1)


grid <- expand.grid(etas = seq(0.01,.3,.001))

grid$loss    <- NA
grid$iter    <- NA

for(i in 1:nrow(grid)){
  
  mod <-  xgb.cv(data                   = dtrain, 
                 params                 = params,
                 objective              = 'binary:logistic',
                 eval_metric            = 'logloss',
                 showsd                 = TRUE,
                 nthread                = 10, 
                 predict                = TRUE,
                 folds                  = myfolds,
                 nrounds                = 10000,
                 eta                    = grid[i,]$etas,
                 early_stopping_rounds  = 100)
  
  logs <- mod$evaluation_log
  
  grid[i,]$iter <- which.min(logs$test_logloss_mean)
  grid[i,]$loss <- min(logs$test_logloss_mean)
  
  print(i)
  
}

grid[which.min(grid$loss),]

head(grid[order(grid$loss),],10)

# Eta = .058, optimized, 64 iteration

mod <-  xgb.cv(data                   = dtrain, 
               params                 = params,
               objective              = 'binary:logistic',
               eval_metric            = 'logloss',
               showsd                 = TRUE,
               nthread                = 10, 
               predict                = TRUE,
               folds                  = myfolds,
               nrounds                = 64,
               eta                    = .058,
               early_stopping_rounds  = 64)

logs <- mod$evaluation_log

plot(logs$train_logloss_mean,type='l',lty=1)
points(logs$test_logloss_mean,type='l',lty=1)


model   <- xgboost(params           = params,
                   objective        = 'binary:logistic',
                   eval_metric      = 'logloss',
                   data             = dtrain,
                   nrounds          = 64,
                   eta              = 0.058)


pr <- predict(model,dtest)


outcome <- data.frame(gender = test$gender, 
                      race   = test$race, 
                      pred   = pr, 
                      out    = out2)


outcome$pred2 <- ifelse(outcome$pred>.5,1,0)

eval(x=outcome,type='brier.m')  # 0.1907071
eval(x=outcome,type='brier.f')  # 0.1585326
eval(x=outcome,type='brier.a')  # 0.1867650
eval(x=outcome,type='fair.m')   # 0.8194426
eval(x=outcome,type='fair.f')   # 0.8049126
auc_roc(preds = outcome$pred,actuals = outcome$out) #0.6993922



################################################################################
################################################################################
#Stage 2:  Max depth and min_child_Weight
################################################################################
################################################################################

params <- list(booster           = "gbtree", 
               gamma             = 0, 
               subsample         = 1, 
               colsample_bytree  = 1,
               max_delta_step    = 0,
               lambda            = 1,
               alpha             = 0,
               scale_pos_weight  = 1,
               num_parallel_tree = 1)


max_dep <- seq(5,15,1)
min_child_weight <- seq(0,7,0.25)

grid <- expand.grid(depth = max_dep, weight = min_child_weight)

grid$loss    <- NA
grid$iter    <- NA

for(i in 1:nrow(grid)){
  
  mod <-  xgb.cv(data                   = dtrain, 
                 params                 = params,
                 objective              = 'binary:logistic',
                 eval_metric            = 'logloss',
                 showsd                 = TRUE,
                 nthread                = 10, 
                 predict                = TRUE,
                 folds                  = myfolds,
                 nrounds                = 64,
                 eta                    = 0.058,
                 max_depth              = grid[i,]$depth, 
                 min_child_weight       = grid[i,]$weight,)
  
  logs <- mod$evaluation_log
  
  grid[i,]$iter <- which.min(logs$test_logloss_mean)
  grid[i,]$loss <- min(logs$test_logloss_mean)
  
  print(i)
  
}

grid[which.min(grid$loss),]

# Depth = 5, weight = 6.75, optimized


model   <- xgboost(params           = params,
                   objective        = 'binary:logistic',
                   eval_metric      = 'logloss',
                   data             = dtrain,
                   nrounds          = 64,
                   eta              = 0.058,
                   max_depth        = 5,
                   min_child_weight = 6.75)


pr <- predict(model,dtest)


outcome <- data.frame(gender = test$gender, 
                      race   = test$race, 
                      pred   = pr, 
                      out    = out2)


outcome$pred2 <- ifelse(outcome$pred>.5,1,0)

eval(x=outcome,type='brier.m')  # 0.1909967
eval(x=outcome,type='brier.f')  # 0.1581383
eval(x=outcome,type='brier.a')  # 0.1869708
eval(x=outcome,type='fair.m')   # 0.8093485
eval(x=outcome,type='fair.f')   # 0.8255149
auc_roc(preds = outcome$pred,actuals = outcome$out) #0.6966467

################################################################################
################################################################################
#Stage 3:  Gamma
################################################################################
################################################################################

params <- list(booster           = "gbtree", 
               subsample         = 1, 
               colsample_bytree  = 1,
               max_delta_step    = 0,
               lambda            = 1,
               alpha             = 0,
               scale_pos_weight  = 1,
               num_parallel_tree = 1)


gamma <- seq(0,1,.01)

grid <- expand.grid(gamma = gamma)

grid$loss    <- NA
grid$iter    <- NA

for(i in 1:nrow(grid)){
  
  mod <-  xgb.cv(data                   = dtrain, 
                 params                 = params,
                 objective              = 'binary:logistic',
                 eval_metric            = 'logloss',
                 showsd                 = TRUE,
                 nthread                = 10, 
                 predict                = TRUE,
                 folds                  = myfolds,
                 nrounds                = 64,
                 eta                    = 0.058,
                 max_depth              = 5,
                 min_child_weight       = 6.75,
                 gamma                  = grid[i,]$gamma)
  
  logs <- mod$evaluation_log
  
  grid[i,]$iter <- which.min(logs$test_logloss_mean)
  grid[i,]$loss <- min(logs$test_logloss_mean)
  
  print(i)
  
}

grid[which.min(grid$loss),]

# Gamma = 0.61, optimized


model   <- xgboost(params           = params,
                   objective        = 'binary:logistic',
                   eval_metric      = 'logloss',
                   data             = dtrain,
                   nrounds          = 64,
                   eta              = 0.058,
                   max_depth        = 5,
                   min_child_weight = 6.75,
                   gamma            = 0.61)


pr <- predict(model,dtest)


outcome <- data.frame(gender = test$gender, 
                      race   = test$race, 
                      pred   = pr, 
                      out    = out2)


outcome$pred2 <- ifelse(outcome$pred>.5,1,0)

eval(x=outcome,type='brier.m')  # 0.1908906
eval(x=outcome,type='brier.f')  # 0.1582531
eval(x=outcome,type='brier.a')  # 0.1868917
eval(x=outcome,type='fair.m')   # 0.8127349
eval(x=outcome,type='fair.f')   # 0.8254023
auc_roc(preds = outcome$pred,actuals = outcome$out) #0.6966417

################################################################################
################################################################################
#Stage 4:  max_delta_step
################################################################################
################################################################################

params <- list(booster           = "gbtree", 
               subsample         = 1, 
               colsample_bytree  = 1,
               lambda            = 1,
               alpha             = 0,
               scale_pos_weight  = 1,
               num_parallel_tree = 1)


delta <- seq(0,10,.1)

grid <- expand.grid(delta = delta)

grid$loss    <- NA
grid$iter    <- NA

for(i in 1:nrow(grid)){
  
  mod <-  xgb.cv(data                   = dtrain, 
                 params                 = params,
                 objective              = 'binary:logistic',
                 eval_metric            = 'logloss',
                 showsd                 = TRUE,
                 nthread                = 10, 
                 predict                = TRUE,
                 folds                  = myfolds,
                 nrounds                = 64,
                 eta                    = 0.058,
                 max_depth              = 5,
                 min_child_weight       = 6.75,
                 gamma                  = 0.61,
                 max_delta_step         = grid[i,]$delta)
  
  logs <- mod$evaluation_log
  
  grid[i,]$iter <- which.min(logs$test_logloss_mean)
  grid[i,]$loss <- min(logs$test_logloss_mean)
  
  print(i)
  
}

grid[which.min(grid$loss),]

# max_delta_step = 1.2, optimized


model   <- xgboost(params                 = params,
                   objective              = 'binary:logistic',
                   eval_metric            = 'logloss',
                   data                   = dtrain,
                   nrounds                = 64,
                   eta                    = 0.058,
                   max_depth              = 5,
                   min_child_weight       = 6.75,
                   gamma                  = 0.61,
                   max_delta_step         = 1.2)


pr <- predict(model,dtest)


outcome <- data.frame(gender = test$gender, 
                      race   = test$race, 
                      pred   = pr, 
                      out    = out2)


outcome$pred2 <- ifelse(outcome$pred>.5,1,0)

eval(x=outcome,type='brier.m')  
eval(x=outcome,type='brier.f')  
eval(x=outcome,type='brier.a')  
eval(x=outcome,type='fair.m')   
eval(x=outcome,type='fair.f')   
auc_roc(preds = outcome$pred,actuals = outcome$out) 
################################################################################
################################################################################
#Stage 5: subsample
################################################################################
################################################################################

params <- list(booster           = "gbtree", 
               colsample_bytree  = 1,
               lambda            = 1,
               alpha             = 0,
               scale_pos_weight  = 1,
               num_parallel_tree = 1)


sample <- seq(0.05,1,.05)

grid <- expand.grid(sample= sample)

grid$loss    <- NA
grid$iter    <- NA

for(i in 1:nrow(grid)){
  
  mod <-  xgb.cv(data                   = dtrain, 
                 params                 = params,
                 objective              = 'binary:logistic',
                 eval_metric            = 'logloss',
                 showsd                 = TRUE,
                 nthread                = 10, 
                 predict                = TRUE,
                 folds                  = myfolds,
                 nrounds                = 64,
                 eta                    = 0.058,
                 max_depth              = 5,
                 min_child_weight       = 6.75,
                 gamma                  = 0.61,
                 max_delta_step         = 1.2,
                 subsample              = grid[i,]$sample)
  
  logs <- mod$evaluation_log
  
  grid[i,]$iter <- which.min(logs$test_logloss_mean)
  grid[i,]$loss <- min(logs$test_logloss_mean)
  
  print(i)
  
}

grid[which.min(grid$loss),]

# subsample = 0.45, optimized


model   <- xgboost(params                 = params,
                   objective              = 'binary:logistic',
                   eval_metric            = 'logloss',
                   data                   = dtrain,
                   nrounds                = 64,
                   eta                    = 0.058,
                   max_depth              = 5,
                   min_child_weight       = 6.75,
                   gamma                  = 0.61,
                   max_delta_step         = 1.2,
                   subsample              = 0.55)


pr <- predict(model,dtest)


outcome <- data.frame(gender = test$gender, 
                      race   = test$race, 
                      pred   = pr, 
                      out    = out2)


outcome$pred2 <- ifelse(outcome$pred>.5,1,0)

eval(x=outcome,type='brier.m')  
eval(x=outcome,type='brier.f')  
eval(x=outcome,type='brier.a')  
eval(x=outcome,type='fair.m')   
eval(x=outcome,type='fair.f')   
auc_roc(preds = outcome$pred,actuals = outcome$out) 


################################################################################
################################################################################
#Stage 6: colsample_bytree
################################################################################
################################################################################

params <- list(booster           = "gbtree", 
               lambda            = 1,
               alpha             = 0,
               scale_pos_weight  = 1,
               num_parallel_tree = 1)


sample <- seq(0.05,1,.05)

grid <- expand.grid(sample= sample)

grid$loss    <- NA
grid$iter    <- NA

for(i in 1:nrow(grid)){
  
  mod <-  xgb.cv(data                   = dtrain, 
                 params                 = params,
                 objective              = 'binary:logistic',
                 eval_metric            = 'logloss',
                 showsd                 = TRUE,
                 nthread                = 20, 
                 predict                = TRUE,
                 folds                  = myfolds,
                 nrounds                = 64,
                 eta                    = 0.058,
                 max_depth              = 5,
                 min_child_weight       = 6.75,
                 gamma                  = 0.61,
                 max_delta_step         = 1.2,
                 subsample              = 0.55,
                 colsample_bytree       = grid[i,]$sample)
  
  logs <- mod$evaluation_log
  
  grid[i,]$iter <- which.min(logs$test_logloss_mean)
  grid[i,]$loss <- min(logs$test_logloss_mean)
  
  print(i)
  
}

grid[which.min(grid$loss),]

# colsample_bytree = 0.8, optimized

model   <- xgboost(params                 = params,
                   objective              = 'binary:logistic',
                   eval_metric            = 'logloss',
                   data                   = dtrain,
                   nrounds                = 64,
                   eta                    = 0.058,
                   max_depth              = 5,
                   min_child_weight       = 6.75,
                   gamma                  = 0.61,
                   max_delta_step         = 1.2,
                   subsample              = 0.55, 
                   colsample_bytree       = 0.8)


pr <- predict(model,dtest)


outcome <- data.frame(gender = test$gender, 
                      race   = test$race, 
                      pred   = pr, 
                      out    = out2)


outcome$pred2 <- ifelse(outcome$pred>.5,1,0)

eval(x=outcome,type='brier.m')  
eval(x=outcome,type='brier.f') 
eval(x=outcome,type='brier.a')  
eval(x=outcome,type='fair.m')   
eval(x=outcome,type='fair.f')   
auc_roc(preds = outcome$pred,actuals = outcome$out) 


################################################################################
################################################################################
#Stage 7: scale_pos_weight
################################################################################
################################################################################

params <- list(booster           = "gbtree", 
               lambda            = 1,
               alpha             = 0,
               num_parallel_tree = 1)


pos <- seq(0,3,.25)

grid <- expand.grid(pos=pos)

grid$loss    <- NA
grid$iter    <- NA

for(i in 1:nrow(grid)){
  
  mod <-  xgb.cv(data                   = dtrain, 
                 params                 = params,
                 objective              = 'binary:logistic',
                 eval_metric            = 'logloss',
                 showsd                 = TRUE,
                 nthread                = 20, 
                 predict                = TRUE,
                 folds                  = myfolds,
                 nrounds                = 64,
                 eta                    = 0.058,
                 max_depth              = 5,
                 min_child_weight       = 6.75,
                 gamma                  = 0.61,
                 max_delta_step         = 1.2,
                 subsample              = 0.55, 
                 colsample_bytree       = 0.8,
                 scale_pos_weight       = grid[i,]$pos)
  
  logs <- mod$evaluation_log
  
  grid[i,]$iter <- which.min(logs$test_logloss_mean)
  grid[i,]$loss <- min(logs$test_logloss_mean)
  
  print(i)
  
}

grid[which.min(grid$loss),]

# scale_pos_Weight = 1, optimized

model   <- xgboost(params                 = params,
                   objective              = 'binary:logistic',
                   eval_metric            = 'logloss',
                   data                   = dtrain,
                   nrounds                = 64,
                   eta                    = 0.058,
                   max_depth              = 5,
                   min_child_weight       = 6.75,
                   gamma                  = 0.61,
                   max_delta_step         = 1.2,
                   subsample              = 0.55, 
                   colsample_bytree       = 0.8,
                   scale_pos_weight       = 1)


pr <- predict(model,dtest)


outcome <- data.frame(gender = test$gender, 
                      race   = test$race, 
                      pred   = pr, 
                      out    = out2)


outcome$pred2 <- ifelse(outcome$pred>.5,1,0)

eval(x=outcome,type='brier.m')  
eval(x=outcome,type='brier.f')  
eval(x=outcome,type='brier.a')  
eval(x=outcome,type='fair.m')   
eval(x=outcome,type='fair.f')   
auc_roc(preds = outcome$pred,actuals = outcome$out)


################################################################################
################################################################################
#Stage 8: num_parallel_tree
################################################################################
################################################################################

params <- list(booster           = "gbtree", 
               lambda            = 1,
               alpha             = 0)


ntree <- seq(1,10,1)

grid <- expand.grid(ntree=ntree)

grid$loss    <- NA
grid$iter    <- NA

for(i in 1:nrow(grid)){
  
  mod <-  xgb.cv(data                   = dtrain, 
                 params                 = params,
                 objective              = 'binary:logistic',
                 eval_metric            = 'logloss',
                 showsd                 = TRUE,
                 nthread                = 20, 
                 predict                = TRUE,
                 folds                  = myfolds,
                 nrounds                = 64,
                 eta                    = 0.058,
                 max_depth              = 5,
                 min_child_weight       = 6.75,
                 gamma                  = 0.61,
                 max_delta_step         = 1.2,
                 subsample              = 0.55, 
                 colsample_bytree       = 0.8,
                 scale_pos_weight       = 1,
                 num_parallel_tree      = grid[i,]$ntree)
  
  logs <- mod$evaluation_log
  
  grid[i,]$iter <- which.min(logs$test_logloss_mean)
  grid[i,]$loss <- min(logs$test_logloss_mean)
  
  print(i)
  
}

grid[which.min(grid$loss),]

# num_parallel_tree = 9, optimized

model   <- xgboost(params                 = params,
                   objective              = 'binary:logistic',
                   eval_metric            = 'logloss',
                   data                   = dtrain,
                   nrounds                = 64,
                   eta                    = 0.058,
                   max_depth              = 5,
                   min_child_weight       = 6.75,
                   gamma                  = 0.61,
                   max_delta_step         = 1.2,
                   subsample              = 0.55, 
                   colsample_bytree       = 0.8,
                   scale_pos_weight       = 1,
                   num_parallel_tree      = 9)


pr <- predict(model,dtest)


outcome <- data.frame(gender = test$gender, 
                      race   = test$race, 
                      pred   = pr, 
                      out    = out2)


outcome$pred2 <- ifelse(outcome$pred>.5,1,0)

eval(x=outcome,type='brier.m')  
eval(x=outcome,type='brier.f')  
eval(x=outcome,type='brier.a')  
eval(x=outcome,type='fair.m')   
eval(x=outcome,type='fair.f')   
auc_roc(preds = outcome$pred,actuals = outcome$out)


################################################################################
################################################################################
#Stage 8: lambda and alpha
################################################################################
################################################################################

params <- list(booster           = "gbtree")


l <- seq(0.1,30,1)
a <- seq(0,6,.5)

grid <- expand.grid(l = l, a = a)

grid$loss    <- NA
grid$iter    <- NA

for(i in 1:nrow(grid)){
  
  mod <-  xgb.cv(data                   = dtrain, 
                 params                 = params,
                 objective              = 'binary:logistic',
                 eval_metric            = 'logloss',
                 showsd                 = TRUE,
                 nthread                = 20, 
                 predict                = TRUE,
                 folds                  = myfolds,
                 nrounds                = 64,
                 eta                    = 0.058,
                 max_depth              = 5,
                 min_child_weight       = 6.75,
                 gamma                  = 0.61,
                 max_delta_step         = 1.2,
                 subsample              = 0.55, 
                 colsample_bytree       = 0.8,
                 scale_pos_weight       = 1,
                 num_parallel_tree      = 9,
                 lambda                 = grid[i,]$l,
                 alpha                  = grid[i,]$a)
  
  logs <- mod$evaluation_log
  
  grid[i,]$iter <- which.min(logs$test_logloss_mean)
  grid[i,]$loss <- min(logs$test_logloss_mean)
  
  print(i)
  
}

grid[which.min(grid$loss),]

# lambda = 2.1 alpha = 0, optimized

model   <- xgboost(params                 = params,
                   objective              = 'binary:logistic',
                   eval_metric            = 'logloss',
                   data                   = dtrain,
                   nrounds                = 64,
                   eta                    = 0.058,
                   max_depth              = 5,
                   min_child_weight       = 6.75,
                   gamma                  = 0.61,
                   max_delta_step         = 1.2,
                   subsample              = 0.55, 
                   colsample_bytree       = 0.8,
                   scale_pos_weight       = 1,
                   num_parallel_tree      = 9,
                   lambda                 = 2.1,
                   alpha                  = 0)


pr <- predict(model,dtest)


outcome <- data.frame(gender = test$gender, 
                      race   = test$race, 
                      pred   = pr, 
                      out    = out2)


outcome$pred2 <- ifelse(outcome$pred>.5,1,0)

eval(x=outcome,type='brier.m')  
eval(x=outcome,type='brier.f')  
eval(x=outcome,type='brier.a')  
eval(x=outcome,type='fair.m')   
eval(x=outcome,type='fair.f')   
auc_roc(preds = outcome$pred,actuals = outcome$out) 


################################################################################
################################################################################
#Stage 8: Reduce learning rate , re-iterate
################################################################################
################################################################################

mod <-  xgb.cv(data                   = dtrain, 
               params                 = params,
               objective              = 'binary:logistic',
               eval_metric            = 'logloss',
               showsd                 = TRUE,
               nthread                = 20, 
               predict                = TRUE,
               folds                  = myfolds,
               nrounds                = 1000,
               eta                    = 0.05,
               max_depth              = 5,
               min_child_weight       = 6.75,
               gamma                  = 0.61,
               max_delta_step         = 1.2,
               subsample              = 0.55, 
               colsample_bytree       = 0.8,
               scale_pos_weight       = 1,
               num_parallel_tree      = 9,
               lambda                 = 2.1,
               alpha                  = 0,
               early_stopping_rounds  = 30)


watchlist <- list(train=dtrain, test=dtest)
model   <- xgb.train(booster                = 'gbtree',
                     objective              = 'binary:logistic',
                     eval_metric            = 'logloss',
                     data                   = dtrain,
                     nrounds                = 111,
                     eta                    = 0.05,
                     max_depth              = 5,
                     min_child_weight       = 6.75,
                     gamma                  = 0.61,
                     max_delta_step         = 1.2,
                     subsample              = 0.55, 
                     colsample_bytree       = 0.8,
                     scale_pos_weight       = 1,
                     num_parallel_tree      = 9,
                     lambda                 = 2.1,
                     alpha                  = 0,
                     early_stopping_rounds  = 30,
                     watchlist              = watchlist)


pr <- predict(model,dtest)


outcome <- data.frame(gender = test$gender, 
                      race   = test$race, 
                      pred   = pr, 
                      out    = out2)


outcome$pred2 <- ifelse(outcome$pred>.5,1,0)

eval(x=outcome,type='brier.m')  # 0.1895345
eval(x=outcome,type='brier.f')  # 0.1564542
eval(x=outcome,type='brier.a')  # 0.1854814
eval(x=outcome,type='fair.m')   # 0.8147695
eval(x=outcome,type='fair.f')   # 0.8150906
auc_roc(preds = outcome$pred,actuals = outcome$out) #0.7035613    

###################################################

imp = xgb.importance(model=model)

xgb.plot.importance(importance_matrix = imp, 
                    top_n = 15,
                    xlim=c(0,.3),
                    xlab='Importance',
                    ylab='Features')

################################################################################

# Save the results for the test sample  


pr <- predict(model,dtest)

outcome <- data.frame(id     = test$ID,
                      gender = test$gender, 
                      race   = test$race, 
                      pred   = pr, 
                      out    = out2)


write.csv(outcome, 
          here('data','test_model1.csv'),
          row.names = FALSE)


################################################################################

head(final_test)

# Save the results for the competition test sample


dtest2  <- xgb.DMatrix(data = data.matrix(final_test[,-1]))

pr2 <- predict(model,dtest2)

outcome2 <- data.frame(id     = final_test$ID,
                       gender = final_test$gender, 
                       race   = final_test$race, 
                       pred   = pr2)

write.csv(outcome2, 
          here('data','comptetition_test_model1.csv'),
          row.names = FALSE)






















