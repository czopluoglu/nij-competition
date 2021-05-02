
require(here)

require(mltools)

require(dplyr)

################################################################################
# Read the datasets

full_data <- read.csv(here('data','NIJ_s_Recidivism_Challenge_Training_Dataset.csv'))

test_submit <- read.csv(here('data','NIJ_s_Recidivism_Challenge_Test_Dataset1.csv'))

################################################################################

# Recode Gender into a dummy variable

full_data$gender   <- ifelse(full_data$Gender=='M',0,1)
test_submit$gender <- ifelse(test_submit$Gender=='M',0,1)

# Recode Race into a dummy variable

full_data$race   <- ifelse(full_data$Race=='WHITE',0,1)
test_submit$race <- ifelse(test_submit$Race=='WHITE',0,1)

# One-hot encoding of Age at Release

full_data$age1 <- ifelse(full_data$Age_at_Release=='18-22',1,0)
full_data$age2 <- ifelse(full_data$Age_at_Release=='23-27',1,0)
full_data$age3 <- ifelse(full_data$Age_at_Release=='28-32',1,0)
full_data$age4 <- ifelse(full_data$Age_at_Release=='33-37',1,0)
full_data$age5 <- ifelse(full_data$Age_at_Release=='38-42',1,0)
full_data$age6 <- ifelse(full_data$Age_at_Release=='43-47',1,0)
full_data$age7 <- ifelse(full_data$Age_at_Release=='48 or older',1,0)


test_submit$age1 <- ifelse(test_submit$Age_at_Release=='18-22',1,0)
test_submit$age2 <- ifelse(test_submit$Age_at_Release=='23-27',1,0)
test_submit$age3 <- ifelse(test_submit$Age_at_Release=='28-32',1,0)
test_submit$age4 <- ifelse(test_submit$Age_at_Release=='33-37',1,0)
test_submit$age5 <- ifelse(test_submit$Age_at_Release=='38-42',1,0)
test_submit$age6 <- ifelse(test_submit$Age_at_Release=='43-47',1,0)
test_submit$age7 <- ifelse(test_submit$Age_at_Release=='48 or older',1,0)

# A numeric age variable by assigning the midpoint of each interval

full_data$age <- recode(.x = full_data$Age_at_Release,
                        '18-22' = 20,
                        '23-27' = 25,
                        '28-32' = 27,
                        '33-37' = 35,
                        '38-42' = 40,
                        '43-47' = 45,
                        '48 or older' = 59)

test_submit$age <- recode(.x = test_submit$Age_at_Release,
                        '18-22' = 20,
                        '23-27' = 25,
                        '28-32' = 27,
                        '33-37' = 35,
                        '38-42' = 40,
                        '43-47' = 45,
                        '48 or older' = 59)


# One-hot encoding of PUMAs

full_data$puma1    <- ifelse(full_data$Residence_PUMA=='1',1,0)
full_data$puma2    <- ifelse(full_data$Residence_PUMA=='2',1,0)
full_data$puma3    <- ifelse(full_data$Residence_PUMA=='3',1,0)
full_data$puma4    <- ifelse(full_data$Residence_PUMA=='4',1,0)
full_data$puma5    <- ifelse(full_data$Residence_PUMA=='5',1,0)
full_data$puma6    <- ifelse(full_data$Residence_PUMA=='6',1,0)
full_data$puma7    <- ifelse(full_data$Residence_PUMA=='7',1,0)
full_data$puma8    <- ifelse(full_data$Residence_PUMA=='8',1,0)
full_data$puma9    <- ifelse(full_data$Residence_PUMA=='9',1,0)
full_data$puma10   <- ifelse(full_data$Residence_PUMA=='10',1,0)
full_data$puma11   <- ifelse(full_data$Residence_PUMA=='11',1,0)
full_data$puma12   <- ifelse(full_data$Residence_PUMA=='12',1,0)
full_data$puma13   <- ifelse(full_data$Residence_PUMA=='13',1,0)
full_data$puma14   <- ifelse(full_data$Residence_PUMA=='14',1,0)
full_data$puma15   <- ifelse(full_data$Residence_PUMA=='15',1,0)
full_data$puma16   <- ifelse(full_data$Residence_PUMA=='16',1,0)
full_data$puma17   <- ifelse(full_data$Residence_PUMA=='17',1,0)
full_data$puma18   <- ifelse(full_data$Residence_PUMA=='18',1,0)
full_data$puma19   <- ifelse(full_data$Residence_PUMA=='19',1,0)
full_data$puma20   <- ifelse(full_data$Residence_PUMA=='20',1,0)
full_data$puma21   <- ifelse(full_data$Residence_PUMA=='21',1,0)
full_data$puma22   <- ifelse(full_data$Residence_PUMA=='22',1,0)
full_data$puma23   <- ifelse(full_data$Residence_PUMA=='23',1,0)
full_data$puma24   <- ifelse(full_data$Residence_PUMA=='24',1,0)
full_data$puma25   <- ifelse(full_data$Residence_PUMA=='25',1,0)



test_submit$puma1  <- ifelse(test_submit$Residence_PUMA=='1',1,0)
test_submit$puma2  <- ifelse(test_submit$Residence_PUMA=='2',1,0)
test_submit$puma3  <- ifelse(test_submit$Residence_PUMA=='3',1,0)
test_submit$puma4  <- ifelse(test_submit$Residence_PUMA=='4',1,0)
test_submit$puma5  <- ifelse(test_submit$Residence_PUMA=='5',1,0)
test_submit$puma6  <- ifelse(test_submit$Residence_PUMA=='6',1,0)
test_submit$puma7  <- ifelse(test_submit$Residence_PUMA=='7',1,0)
test_submit$puma8  <- ifelse(test_submit$Residence_PUMA=='8',1,0)
test_submit$puma9  <- ifelse(test_submit$Residence_PUMA=='9',1,0)
test_submit$puma10 <- ifelse(test_submit$Residence_PUMA=='10',1,0)
test_submit$puma11 <- ifelse(test_submit$Residence_PUMA=='11',1,0)
test_submit$puma12 <- ifelse(test_submit$Residence_PUMA=='12',1,0)
test_submit$puma13 <- ifelse(test_submit$Residence_PUMA=='13',1,0)
test_submit$puma14 <- ifelse(test_submit$Residence_PUMA=='14',1,0)
test_submit$puma15 <- ifelse(test_submit$Residence_PUMA=='15',1,0)
test_submit$puma16 <- ifelse(test_submit$Residence_PUMA=='16',1,0)
test_submit$puma17 <- ifelse(test_submit$Residence_PUMA=='17',1,0)
test_submit$puma18 <- ifelse(test_submit$Residence_PUMA=='18',1,0)
test_submit$puma19 <- ifelse(test_submit$Residence_PUMA=='19',1,0)
test_submit$puma20 <- ifelse(test_submit$Residence_PUMA=='20',1,0)
test_submit$puma21 <- ifelse(test_submit$Residence_PUMA=='21',1,0)
test_submit$puma22 <- ifelse(test_submit$Residence_PUMA=='22',1,0)
test_submit$puma23 <- ifelse(test_submit$Residence_PUMA=='23',1,0)
test_submit$puma24 <- ifelse(test_submit$Residence_PUMA=='24',1,0)
test_submit$puma25 <- ifelse(test_submit$Residence_PUMA=='25',1,0)

# Recode Gang_Affiliated into a dummy variable

full_data$gang <- recode(.x = full_data$Gang_Affiliated,
                         'true' = 1,
                         'false' = 0)


test_submit$gang <- recode(.x = test_submit$Gang_Affiliated,
                         'true' = 1,
                         'false' = 0)

# Supervision_Risk_Score_First

full_data$risk1 <- full_data$Supervision_Risk_Score_First

test_submit$risk1 <- test_submit$Supervision_Risk_Score_First

# One hot encoding of Supervision_Level_First

full_data$slevel1 <- recode(.x = full_data$Supervision_Level_First,
                            'High' = 1,
                            'Specialized' = 0,
                            'Standard' = 0)

full_data$slevel2 <- recode(.x = full_data$Supervision_Level_First,
                            'High' = 0,
                            'Specialized' = 1,
                            'Standard' = 0)


full_data$slevel3 <- recode(.x = full_data$Supervision_Level_First,
                            'High' = 0,
                            'Specialized' = 0,
                            'Standard' = 1)



test_submit$slevel1 <- recode(.x = test_submit$Supervision_Level_First,
                            'High' = 1,
                            'Specialized' = 0,
                            'Standard' = 0)

test_submit$slevel2 <- recode(.x = test_submit$Supervision_Level_First,
                            'High' = 0,
                            'Specialized' = 1,
                            'Standard' = 0)


test_submit$slevel3 <- recode(.x = test_submit$Supervision_Level_First,
                            'High' = 0,
                            'Specialized' = 0,
                            'Standard' = 1)

# Education Level

full_data$educ1 <- ifelse(full_data$Education_Level=='At least some college',1,0)
full_data$educ2 <- ifelse(full_data$Education_Level=='High School Diploma',1,0)
full_data$educ3 <- ifelse(full_data$Education_Level=='Less than HS diploma',1,0)
full_data$educ  <- recode(.x = full_data$Education_Level,
                          'At least some college' = 1,
                          'High School Diploma' = 2,
                          'Less than HS diploma' = 3)

test_submit$educ1 <- ifelse(test_submit$Education_Level=='At least some college',1,0)
test_submit$educ2 <- ifelse(test_submit$Education_Level=='High School Diploma',1,0)
test_submit$educ3 <- ifelse(test_submit$Education_Level=='Less than HS diploma',1,0)
test_submit$educ  <- recode(.x = test_submit$Education_Level,
                          'At least some college' = 1,
                          'High School Diploma' = 2,
                          'Less than HS diploma' = 3)

# Dependents

full_data$dep1 <- ifelse(full_data$Dependents=="0",1,0)
full_data$dep2 <- ifelse(full_data$Dependents=="1",1,0)
full_data$dep3 <- ifelse(full_data$Dependents=="2",1,0)
full_data$dep4 <- ifelse(full_data$Dependents=="3 or more",1,0)
full_data$dep  <- recode(.x = full_data$Dependents,
                          '0' = 1,
                          '1' = 2,
                          '2' = 3,
                          '3 or more'=4)


test_submit$dep1 <- ifelse(test_submit$Dependents=="0",1,0)
test_submit$dep2 <- ifelse(test_submit$Dependents=="1",1,0)
test_submit$dep3 <- ifelse(test_submit$Dependents=="2",1,0)
test_submit$dep4 <- ifelse(test_submit$Dependents=="3 or more",1,0)
test_submit$dep  <- recode(.x = test_submit$Dependents,
                         '0' = 1,
                         '1' = 2,
                         '2' = 3,
                         '3 or more'=4)



# One hot encoding of Prison Offense

full_data$off1 <- recode(.x = full_data$Prison_Offense,
                         'Drug' = 1,
                         'Other' = 0,
                         'Property' = 0,
                         'Violent/Non-Sex' = 0,
                         'Violent/Sex'=0)

full_data$off2 <- recode(.x = full_data$Prison_Offense,
                         'Drug' = 0,
                         'Other' = 1,
                         'Property' = 0,
                         'Violent/Non-Sex' = 0,
                         'Violent/Sex'=0)

full_data$off3 <- recode(.x = full_data$Prison_Offense,
                         'Drug' = 0,
                         'Other' = 0,
                         'Property' = 1,
                         'Violent/Non-Sex' = 0,
                         'Violent/Sex'=0)

full_data$off4 <- recode(.x = full_data$Prison_Offense,
                         'Drug' = 0,
                         'Other' = 0,
                         'Property' = 0,
                         'Violent/Non-Sex' = 1,
                         'Violent/Sex'=0)

full_data$off5 <- recode(.x = full_data$Prison_Offense,
                         'Drug' = 0,
                         'Other' = 0,
                         'Property' = 0,
                         'Violent/Non-Sex' = 0,
                         'Violent/Sex'=1)




test_submit$off1 <- recode(.x = test_submit$Prison_Offense,
                         'Drug' = 1,
                         'Other' = 0,
                         'Property' = 0,
                         'Violent/Non-Sex' = 0,
                         'Violent/Sex'=0)

test_submit$off2 <- recode(.x = test_submit$Prison_Offense,
                         'Drug' = 0,
                         'Other' = 1,
                         'Property' = 0,
                         'Violent/Non-Sex' = 0,
                         'Violent/Sex'=0)

test_submit$off3 <- recode(.x = test_submit$Prison_Offense,
                         'Drug' = 0,
                         'Other' = 0,
                         'Property' = 1,
                         'Violent/Non-Sex' = 0,
                         'Violent/Sex'=0)

test_submit$off4 <- recode(.x = test_submit$Prison_Offense,
                         'Drug' = 0,
                         'Other' = 0,
                         'Property' = 0,
                         'Violent/Non-Sex' = 1,
                         'Violent/Sex'=0)

test_submit$off5 <- recode(.x = test_submit$Prison_Offense,
                         'Drug' = 0,
                         'Other' = 0,
                         'Property' = 0,
                         'Violent/Non-Sex' = 0,
                         'Violent/Sex'=1)




# One hot encoding of Prison years

full_data$year1 <- recode(.x = full_data$Prison_Years,
                         '1-2 years' = 1,
                         'Greater than 2 to 3 years'= 0,
                         'Less than 1 year' = 0,
                         'More than 3 years' = 0)

full_data$year2 <- recode(.x = full_data$Prison_Years,
                          '1-2 years' = 0,
                          'Greater than 2 to 3 years' = 1,
                          'Less than 1 year' = 0,
                          'More than 3 years' = 0)

full_data$year3 <- recode(.x = full_data$Prison_Years,
                          '1-2 years' = 0,
                          'Greater than 2 to 3 years' = 0,
                          'Less than 1 year' = 1,
                          'More than 3 years' = 0)

full_data$year4 <- recode(.x = full_data$Prison_Years,
                          '1-2 years' = 0,
                          'Greater than 2 to 3 years' = 0,
                          'Less than 1 year' = 0,
                          'More than 3 years' = 1)




test_submit$year1 <- recode(.x = test_submit$Prison_Years,
                          '1-2 years' = 1,
                          'Greater than 2 to 3 years'= 0,
                          'Less than 1 year' = 0,
                          'More than 3 years' = 0)

test_submit$year2 <- recode(.x = test_submit$Prison_Years,
                          '1-2 years' = 0,
                          'Greater than 2 to 3 years' = 1,
                          'Less than 1 year' = 0,
                          'More than 3 years' = 0)

test_submit$year3 <- recode(.x = test_submit$Prison_Years,
                          '1-2 years' = 0,
                          'Greater than 2 to 3 years' = 0,
                          'Less than 1 year' = 1,
                          'More than 3 years' = 0)

test_submit$year4 <- recode(.x = test_submit$Prison_Years,
                          '1-2 years' = 0,
                          'Greater than 2 to 3 years'= 0,
                          'Less than 1 year' = 0,
                          'More than 3 years' = 1)


# Prior_Arrest_Episodes_Felony

full_data$felony <- as.numeric(
                        ifelse(full_data$Prior_Arrest_Episodes_Felony=='10 or more',
                               10,full_data$Prior_Arrest_Episodes_Felony))

full_data$felony_cat <- ifelse(full_data$Prior_Arrest_Episodes_Felony=='10 or more',1,0)

test_submit$felony <- as.numeric(
  ifelse(test_submit$Prior_Arrest_Episodes_Felony=='10 or more',
         10,test_submit$Prior_Arrest_Episodes_Felony))

test_submit$felony_cat <- ifelse(test_submit$Prior_Arrest_Episodes_Felony=='10 or more',1,0)

# Prior_Arrest_Episodes_Misd

full_data$misd <- as.numeric(
  ifelse(full_data$Prior_Arrest_Episodes_Misd=='6 or more',
         6,full_data$Prior_Arrest_Episodes_Misd))

full_data$misd_cat <- ifelse(full_data$Prior_Arrest_Episodes_Misd=='6 or more',1,0)
      

test_submit$misd <- as.numeric(
  ifelse(test_submit$Prior_Arrest_Episodes_Misd=='6 or more',
         6,test_submit$Prior_Arrest_Episodes_Misd))

test_submit$misd_cat <- ifelse(test_submit$Prior_Arrest_Episodes_Misd=='6 or more',1,0)


# Prior_Arrest_Episodes_Violent

full_data$viol <- as.numeric(
  ifelse(full_data$Prior_Arrest_Episodes_Violent=='3 or more',
         3,full_data$Prior_Arrest_Episodes_Violent))

full_data$viol_cat <- ifelse(full_data$Prior_Arrest_Episodes_Violent=='3 or more',1,0)


test_submit$viol <- as.numeric(
  ifelse(test_submit$Prior_Arrest_Episodes_Violent=='3 or more',
         3,test_submit$Prior_Arrest_Episodes_Violent))

test_submit$viol_cat <- ifelse(test_submit$Prior_Arrest_Episodes_Violent=='3 or more',1,0)


# Prior_Arrest_Episodes_Property

full_data$prop <- as.numeric(
  ifelse(full_data$Prior_Arrest_Episodes_Property=='5 or more',
         5,full_data$Prior_Arrest_Episodes_Property))

full_data$prop_cat <- ifelse(full_data$Prior_Arrest_Episodes_Property=='5 or more',1,0)


test_submit$prop <- as.numeric(
  ifelse(test_submit$Prior_Arrest_Episodes_Property=='5 or more',
         5,test_submit$Prior_Arrest_Episodes_Property))

test_submit$prop_cat <- ifelse(test_submit$Prior_Arrest_Episodes_Property=='5 or more',1,0)

# Prior_Arrest_Episodes_Drug

full_data$drug <- as.numeric(
  ifelse(full_data$Prior_Arrest_Episodes_Drug=='5 or more',
         5,full_data$Prior_Arrest_Episodes_Drug))

full_data$drug_cat <- ifelse(full_data$Prior_Arrest_Episodes_Drug=='5 or more',1,0)


test_submit$drug <- as.numeric(
  ifelse(test_submit$Prior_Arrest_Episodes_Drug=='5 or more',
         5,test_submit$Prior_Arrest_Episodes_Drug))

test_submit$drug_cat <- ifelse(test_submit$Prior_Arrest_Episodes_Drug=='5 or more',1,0)

# Prior_Arrest_Episodes_DVCharges

full_data$dv <- ifelse(full_data$Prior_Arrest_Episodes_DVCharges=='true',1,0)
test_submit$dv <- ifelse(test_submit$Prior_Arrest_Episodes_DVCharges=='true',1,0)

# Prior_Arrest_Episodes_GunCharges

full_data$gun <- ifelse(full_data$Prior_Arrest_Episodes_GunCharges=='true',1,0)
test_submit$gun <- ifelse(test_submit$Prior_Arrest_Episodes_GunCharges=='true',1,0)


# Prior_Conviction_Episodes_Felony

full_data$cfelony <- as.numeric(
  ifelse(full_data$Prior_Conviction_Episodes_Felony=='3 or more',
         3,full_data$Prior_Conviction_Episodes_Felony))

full_data$cfelony_cat <- ifelse(full_data$Prior_Conviction_Episodes_Felony=='3 or more',1,0)

test_submit$cfelony <- as.numeric(
  ifelse(test_submit$Prior_Conviction_Episodes_Felony=='3 or more',
         3,test_submit$Prior_Conviction_Episodes_Felony))

test_submit$cfelony_cat <- ifelse(test_submit$Prior_Conviction_Episodes_Felony=='3 or more',1,0)


# Prior_Conviction_Episodes_Misd

full_data$cmisd <- as.numeric(
  ifelse(full_data$Prior_Conviction_Episodes_Misd=='4 or more',
         4,full_data$Prior_Conviction_Episodes_Misd))

full_data$cmisd_cat <- ifelse(full_data$Prior_Conviction_Episodes_Misd=='4 or more',1,0)

test_submit$cmisd <- as.numeric(
  ifelse(test_submit$Prior_Conviction_Episodes_Misd=='4 or more',
         4,test_submit$Prior_Conviction_Episodes_Misd))

test_submit$cmisd_cat <- ifelse(test_submit$Prior_Conviction_Episodes_Misd=='4 or more',1,0)

# Prior_Conviction_Episodes_Viol

full_data$cviol <- ifelse(full_data$Prior_Conviction_Episodes_Viol=='true',1,0)
test_submit$cviol <- ifelse(test_submit$Prior_Conviction_Episodes_Viol=='true',1,0)

# Prior_Conviction_Episodes_Prop

full_data$cprop <- as.numeric(
  ifelse(full_data$Prior_Conviction_Episodes_Prop=='3 or more',
         3,full_data$Prior_Conviction_Episodes_Prop))

full_data$cprop_cat <- ifelse(full_data$Prior_Conviction_Episodes_Prop=='3 or more',1,0)


test_submit$cprop <- as.numeric(
  ifelse(test_submit$Prior_Conviction_Episodes_Prop=='3 or more',
         3,test_submit$Prior_Conviction_Episodes_Prop))

test_submit$cprop_cat <- ifelse(test_submit$Prior_Conviction_Episodes_Prop=='3 or more',1,0)

# Prior_Conviction_Episodes_Drug

full_data$cdrug <- as.numeric(
  ifelse(full_data$Prior_Conviction_Episodes_Drug=='2 or more',
         2,full_data$Prior_Conviction_Episodes_Drug))

full_data$cdrug_cat <- ifelse(full_data$Prior_Conviction_Episodes_Drug=='2 or more',1,0)


test_submit$cdrug <- as.numeric(
  ifelse(test_submit$Prior_Conviction_Episodes_Drug=='2 or more',
         2,test_submit$Prior_Conviction_Episodes_Drug))

test_submit$cdrug_cat <- ifelse(test_submit$Prior_Conviction_Episodes_Drug=='2 or more',1,0)

# X_v1

full_data$Xv1 <- as.numeric(ifelse(full_data$X_v1=='5 or more',5,full_data$X_v1))

full_data$Xv1_cat <- ifelse(full_data$X_v1=='5 or more',1,0)


test_submit$Xv1 <- as.numeric(ifelse(test_submit$X_v1=='5 or more',5,test_submit$X_v1))

test_submit$Xv1_cat <- ifelse(test_submit$X_v1=='5 or more',1,0)

# X_v2

full_data$Xv2   <- ifelse(full_data$X_v2=='true',1,0)
test_submit$Xv2 <- ifelse(test_submit$X_v2=='true',1,0)

# X_v3

full_data$Xv3    <- ifelse(full_data$X_v3=='true',1,0)
test_submit$Xv3  <- ifelse(test_submit$X_v3=='true',1,0)

# X_v4

full_data$Xv4    <- ifelse(full_data$X_v4=='true',1,0)
test_submit$Xv4  <- ifelse(test_submit$X_v4=='true',1,0)

# Prior_Revocations_Parole

full_data$prior_parole    <- ifelse(full_data$Prior_Revocations_Parole=='true',1,0)
test_submit$prior_parole  <- ifelse(test_submit$Prior_Revocations_Parole=='true',1,0)

# Prior_Revocations_Probation     

full_data$prior_prob    <- ifelse(full_data$Prior_Revocations_Probation=='true',1,0)
test_submit$prior_prob  <- ifelse(test_submit$Prior_Revocations_Probation=='true',1,0)

# Condition_MH_SA                  

full_data$mhsa    <- ifelse(full_data$Condition_MH_SA=='true',1,0)
test_submit$mhsa  <- ifelse(test_submit$Condition_MH_SA=='true',1,0)

# Condition_Cog_Ed

full_data$coged    <- ifelse(full_data$Condition_Cog_Ed=='true',1,0)
test_submit$coged  <- ifelse(test_submit$Condition_Cog_Ed=='true',1,0)

# Condition_Other

full_data$cother    <- ifelse(full_data$Condition_Other=='true',1,0)
test_submit$cother  <- ifelse(test_submit$Condition_Other=='true',1,0)


# Violations_ElectronicMonitoring

full_data$viol_monit    <- ifelse(full_data$Violations_ElectronicMonitoring=='true',1,0)

  #test_submit$viol_monit  <- ifelse(test_submit$Violations_ElectronicMonitoring=='true',1,0)

# Violations_Instruction

full_data$viol_inst    <- ifelse(full_data$Violations_Instruction=='true',1,0)

  #test_submit$viol_inst  <- ifelse(test_submit$Violations_Instruction=='true',1,0)

# Violations_FailToReport

full_data$viol_rep  <- ifelse(full_data$Violations_FailToReport=='true',1,0)

  #test_submit$viol_rep  <- ifelse(test_submit$Violations_FailToReport=='true',1,0)

# Violations_MoveWithoutPermission

full_data$viol_move  <- ifelse(full_data$Violations_MoveWithoutPermission=='true',1,0)

  #test_submit$viol_move  <- ifelse(test_submit$Violations_MoveWithoutPermission=='true',1,0)


# Delinquency_Reports

full_data$delinq <- as.numeric(
  ifelse(full_data$Delinquency_Reports=='4 or more',
         4,full_data$Delinquency_Reports))

full_data$delinq_cat <- ifelse(full_data$Delinquency_Reports=='4 or more',1,0)

  #test_submit$delinq <- as.numeric(
  #  ifelse(test_submit$Delinquency_Reports=='4 or more',
  #         4,test_submit$Delinquency_Reports))
  #
  #test_submit$delinq_cat <- ifelse(test_submit$Delinquency_Reports=='4 or more',1,0)


# Program_Attendances

full_data$prat <- as.numeric(
  ifelse(full_data$Program_Attendances=='10 or more',
         10,full_data$Program_Attendances))

full_data$prat_cat <- ifelse(full_data$Program_Attendances=='10 or more',1,0)

  # test_submit$prat <- as.numeric(
  #  ifelse(test_submit$Program_Attendances=='10 or more',
  #         10,test_submit$Program_Attendances))
  #
  # test_submit$prat_cat <- ifelse(test_submit$Program_Attendances=='10 or more',1,0)


# Program_UnexcusedAbsences

full_data$prun <- as.numeric(
  ifelse(full_data$Program_UnexcusedAbsences=='3 or more',
         3,full_data$Program_UnexcusedAbsences))

full_data$prun_cat <- ifelse(full_data$Program_UnexcusedAbsences=='3 or more',1,0)

# test_submit$prun <- as.numeric(
#  ifelse(test_submit$Program_UnexcusedAbsences=='3 or more',
#         3,test_submit$Program_UnexcusedAbsences))
#
# test_submit$prun_cat <- ifelse(test_submit$Program_UnexcusedAbsences=='3 or more',1,0)

# Residence_Changes

full_data$resch <- as.numeric(
  ifelse(full_data$Residence_Changes=='3 or more',
         3,full_data$Residence_Changes))

full_data$resch_cat <- ifelse(full_data$Residence_Changes=='3 or more',1,0)

# test_submit$resch <- as.numeric(
#  ifelse(test_submit$Residence_Changes=='3 or more',
#         3,test_submit$Residence_Changes))
#
# test_submit$resch_cat <- ifelse(test_submit$Residence_Changes=='3 or more',1,0)


# Avg_Days_per_DrugTest           

full_data$avg_drug <- log(full_data$Avg_Days_per_DrugTest)

  #test_submit$avg_drug <- log(test_submit$Avg_Days_per_DrugTest)

# DrugTests_THC_Positive
  
full_data$thc <- full_data$DrugTests_THC_Positive

  # test_submit$thc <- test_submit$DrugTests_THC_Positive

# DrugTests_Cocaine_Positive

full_data$coc <- full_data$DrugTests_Cocaine_Positive

  # test_submit$coc <- test_submit$DrugTests_Cocaine_Positive

# DrugTests_Meth_Positive

full_data$meth <- full_data$DrugTests_Meth_Positive

  # test_submit$meth <- test_submit$DrugTests_Meth_Positive
 
# DrugTests_Other_Positive
  
full_data$other <- full_data$DrugTests_Other_Positive

  # test_submit$other <- test_submit$DrugTests_Other_Positive

# Percent_Days_Employed

full_data$pemployed <- full_data$Percent_Days_Employed

  #test_submit$pemployed <- test_submit$Percent_Days_Employed

# Jobs_Per_Year

full_data$jobs <- full_data$Jobs_Per_Year

  # test_submit$jobs <- test_submit$Jobs_Per_Year

# Employment_Exempt

full_data$employed <- ifelse(full_data$Employment_Exempt=='true',1,0)
  
  # test_submit$employed <- ifelse(test_submit$Employment_Exempt=='true',1,0)
  
# Recidivism_Within_3years

full_data$w3 <- ifelse(full_data$Recidivism_Within_3years=='true',1,0)

# Recidivism_Arrest_Year1 

full_data$y1 <- ifelse(full_data$Recidivism_Arrest_Year1=='true',1,0)

# Recidivism_Arrest_Year2

full_data$y2 <- ifelse(full_data$Recidivism_Arrest_Year2=='true',1,0)

# Recidivism_Arrest_Year3

full_data$y3 <- ifelse(full_data$Recidivism_Arrest_Year3=='true',1,0)

###############################################################################


corr <- cor(full_data[,c('felony','misd','viol','prop','drug','dv','gun',
                         'cfelony','cmisd','cviol','cprop','cdrug',
                         'Xv1','Xv2','Xv3','Xv4')])

plot(eigen(corr)$values)


principal(corr,nfactors=4,rotate='promax')

# Group 1: felony, misd, prop, cfelony, cmisd, cprop,Xv1,Xv2
# Group 2: drug, cdrug
# Group 3: viol, dv, cviol,Xv3
# Group 4: gun, Xv4


full_data$comp1 <- rowMeans(full_data[,c('felony','misd','prop','cfelony','cmisd',
                                     'cprop','Xv1','Xv2')],na.rm=TRUE)

full_data$comp2 <- rowMeans(full_data[,c('drug','cdrug')],na.rm=TRUE)
full_data$comp3 <- rowMeans(full_data[,c('viol','dv','Xv3')],na.rm=TRUE)
full_data$comp4 <- rowMeans(full_data[,c('gun','Xv4')],na.rm=TRUE)


test_submit$comp1 <- rowMeans(test_submit[,c('felony','misd','prop','cfelony','cmisd',
                                         'cprop','Xv1','Xv2')],na.rm=TRUE)

test_submit$comp2 <- rowMeans(test_submit[,c('drug','cdrug')],na.rm=TRUE)
test_submit$comp3 <- rowMeans(test_submit[,c('viol','dv','Xv3')],na.rm=TRUE)
test_submit$comp4 <- rowMeans(test_submit[,c('gun','Xv4')],na.rm=TRUE)


###############################################################################


final_test <- test_submit[,c(1,34:126)]

train_ <- full_data[,colnames(full_data)%in%colnames(final_test)]

out <- full_data$y1





















