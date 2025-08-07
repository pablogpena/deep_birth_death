# -*- coding: utf-8 -*-
# # Workspace configuration

no_fossil_path <- "/workspace/phylo_estimation/simulations/674_10k/"

# # Simulation configuration

n_sim <- 10000
n_tips <- 674
complete = FALSE 

# ## Simulations configuration

# +
switch(args[1], 
    BD={
        get_r0 <- function() {return(runif(1, 0.01, 4))}
        get_a0 <- function() {return(runif(1, 0.01, 0.5))} 
        get_frac0 <- function() {return(1)} 
        
        get_r1 <- function(r0) {return (r0)}
        get_a1 <- function(a0) {return (a0)}
        get_frac1 <- function(frac0) {return (frac0)}
        get_time <- function() {return(0)}
        
        time_limit <- 0
    },
    
    HE={
        get_r0 <- function() {return(runif(1, 0.01, 4))}
        get_a0 <- function() {return(runif(1, 0.8, 0.9))} 
        get_frac0 <- function() {return(1)} 
        
        get_r1 <- function(r0) {return (r0)}
        get_a1 <- function(a0) {return (a0)}
        get_frac1 <- function(frac0) {return (frac0)}
        get_time <- function() {return(0)}
        
        time_limit <- 0
    }, 
       
    ME_rho={
        get_r0 <- function() {return(runif(1, 0.1, 1))}
        get_a0 <- function() {return(runif(1, 0.3, 0.8))} 
        get_frac0 <- function() {return(1)} 
        
        get_r1 <- function(r0) {return (r0)}
        get_a1 <- function(a0) {return (a0)}
        get_frac1 <- function(frac0) {return(runif(1, 0.1, 0.3))}
        get_time <- function() {return(runif(1, 3, 20))}
        
        time_limit <- 5
    },    
       
    SR={
        get_r0 <- function() {return(runif(1, 0.25, 1.99))}
        get_a0 <- function() {return(runif(1, 0.05, 0.5))} 
        get_frac0 <- function() {return(1)} 
        
        get_r1 <- function(r0) {runif(1, 0.01, 0.1)}
        get_a1 <- function(a0) {runif(1, 0.55, 0.95)}
        get_frac1 <- function(frac0) {return (frac0)}
        get_time <- function() {return(runif(1, 3, 20))}
        
        time_limit <- 5
    },
       
    WW={
        get_r0 <- function() {return(runif(1, -0.2, -0.01))}
        get_a0 <- function() {return(runif(1, 1.3, 2))} 
        get_frac0 <- function() {return(1)} 
        
        get_r1 <- function(r0) {runif(1, 0.5, 1.5)}
        get_a1 <- function(a0) {runif(1, 0.25, 0.6)}
        get_frac1 <- function(frac0) {return (frac0)}
        get_time <- function() {return(runif(1, 3, 20))}
        
        time_limit <- 5
    }

)
