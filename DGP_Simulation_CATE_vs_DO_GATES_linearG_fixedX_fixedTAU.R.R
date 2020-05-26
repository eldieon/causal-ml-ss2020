### This generates the Simulation Data as a dataframe
# by Daniel Jacob (daniel.jacob@hu-berlin.de) 

# Arguments to specify are: 

# N = Number of observations (real number)
# k = Number of covariates (real number)
# random_d = treatment assignment: (Either T for random assignment or F for confounding on X)
# theta = treatment effect: (Either real number for only one theta, or "binary" {0.1,0.3} or "con" for continuous values (0.1,0.3))
# var = Size of the variance (Noise-level)

#Required Packages
if(!require("clusterGeneration")) install.packages("clusterGeneration"); library("clusterGeneration")
if(!require("mvtnorm")) install.packages("mvtnorm"); library("mvtnorm")
if(!require("ggplot2")) install.packages("ggplot2"); library("ggplot2")
if(!require("reshape2")) install.packages("reshape2"); library("reshape2")
if(!require("cowplot")) install.packages("cowplot"); library("cowplot")



N = 2000
k = 20
b = 1 / (1:k)
# = Generate covariance matrix of z = #
sigma1 <- genPositiveDefMat(k, "unifcorrmat")$Sigma
sigma <- cov2cor(sigma1)


# got dimensions from the number of observattions (2000), and the size of the covariance matrix.
# provide the density function and a random number generator for the multivariate normal distribution 
# with mean equal to mean and covariance matrix sigma.

z_fix <- rmvnorm(N, sigma = sigma) # = Generate z 

## takes multivariate normal dist, with mean = 0 and covariance sigma (k*k)


### Options for theta

###theta_s for singular theta. => as a numeric vector of 2000 obs, gives the theta as a normal random with mean 0 and var 0.5
  #theta_s <- as.vector(z_fix[,1] + (z_fix[,2]>0) + rnorm(N,0,0.5))

###con_lin = continuous and linear => takes theta_s and 
  #theta_con_lin <- (theta_s - min(theta_s)) * (1 - 0.1) / (max(theta_s) - min(theta_s)) +
   # 0.1

    #theta_s <- as.vector(sin(z_fix %*% b)+z_fix[,(5+k/2)])
    #theta_con_non <- (theta_s - min(theta_s)) * (1 - 0.1) / (max(theta_s) - min(theta_s)) +
     # 0.1

theta_con_lin <- as.vector(z_fix[,1] + (z_fix[,2]>0) + rnorm(N,0,0.5)) #is this not singular theta?
theta_con_non <- as.vector(sin(z_fix %*% b)+z_fix[,(5+k/2)]) #nonlinear because of sine function. 


datagen <- function(N,y,k,random_d,theta,var) {
  
  z <- z_fix
  
  ### Options for D (m_(X))
  if (random_d == T) {
    d <- rep(c(0, 1), length.out = N)
  } else 
    if(random_d =="imbalanced"){
     d <-  as.numeric(rbinom(N,prob=0.2,size=1))
     
    }
    else
      
      
    if(random_d == "linear"){
      d_prop <- pnorm( z[,k/2] + z[,2] + z[,k/4] - z[,8]) # D is dependent on Za
      d <- as.numeric(rbinom(N, prob = d_prop, size = 1))
    }
     else 
       if(random_d == "interaction"){
    d_prop <- pnorm((z %*% b) + z[,k/2] + z[,2] + z[,k/4]*z[,8]) # D is dependent on Za
    d <- as.numeric(rbinom(N, prob = d_prop, size = 1))
       }
  else{
    d_prop <- pnorm((z %*% b) + sin(z[,k/2]) + z[,2] + cos(z[,k/4]*z[,8])) # D is dependent on Za
    d <- as.numeric(rbinom(N, prob = d_prop, size = 1))
    
  }
  
  if(theta=="con_lin"){
    theta <- theta_con_lin}
  else{theta <- theta_con_non}
  
  
  
  g <- as.vector(z[,k/10] + z[,k/2] + z[,k/4]*z[,k/10])
  
  
  if(y=="binary") {
    y1 <- theta * d + g 
    y1.1 <- rbinom(N,prob=pnorm(scale(y1)),size=1)
    #y1.1 <- (y1 - min(y1)) * (1) / (max(y1) - min(y1)) + 0
    #y <-  rbinom(N,prob=y1.1,size=1)
    y <- y1.1
  } else {y <- theta * d + g + rnorm(N,0,var)}
  
  data <- as.data.frame(y)
  data <- cbind(data, theta, d, z)
  colnames(data) <- c("y", "theta", "d", c(paste0("V", 1:k)))
  
  return(data)
}


### Example

###THETA LINEAR AND THETA NON LINEAR ARE THE SAME??

dataset <- datagen(y="con",N = 500, k = 20, random_d = "linear", theta = "con_lin", var = 1)

dataset_non_linear <- datagen(y="con",N = 500, k = 20, random_d = "linear", theta = "con_non", var = 1) 
#summary(dataset)
#str(dataset)

theta_linear <- dataset$theta
theta_non_linear <- dataset_non_linear$theta

theta_all <- cbind(theta_linear,theta_non_linear)
colnames(theta_all) <- c("theta linear","theta non-linear")
mm <- melt(theta_all)

ggplot(mm, aes(x=value)) + 
 geom_density() +
  theme_cowplot() +
  facet_wrap( ~ Var2, scales="free", ncol=2)  + # Facet wrap with common scales 
  theme(legend.position="bottom", legend.justification = 'center')
