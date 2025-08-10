# -*- coding: utf-8 -*-
library(DDD)
library(ape)
library(readr)

AIC <- function(Lik, k) {
    AIC <- 2*k -2*(Lik) #Lik here is positive, as DDD returns the loglikelihood in positive 
    return(AIC)
}

empiric_path <- "/workspace/deep_birth_death/empirical/"
tree_names <- c("eucalypts.nwk", "conifers.nwk", "cetaceans.nwk")
out_path <- "/workspace/deep_birth_death//MLE/inference_data/empiric"

#Create empty CSV for saving results line by line 
write.table(
        data.frame(
        name = character(),
        likelihood = numeric(),
        AIC = numeric(),
        estimated_lambda = numeric()
        ),
                   
        file = file.path(out_path, paste0("DDD_empiric_inference.csv")),
        row.names = FALSE, 
        col.names = TRUE,
        quote=FALSE,
        sep=',')

# +
for (tree in tree_names) {
    
    #Load trees and data
    tr <- read.tree(paste0(empiric_path, tree))
    n_tips <- Ntip(tr)  
    k <- n_tips + 1
    brts <- ape::branching.times(tr)
    
    missnumspec <- 0
    ddmodel <- 1 #Linear model 
    
    # We only want to estimate inital lambda
    initparsopt <- DDD:::initparsoptdefault(ddmodel, brts, missnumspec)[1] 
    idparsopt <- c(1)    
    
    #We fix paremeters mu and k 
    idparsopt <- c(1)           
    idparsfix <- c(2, 3)         
    parsfix <- c(0, k)  
    
    res <- 10 * (1 + length(brts) + missnumspec)
    
    dd_inf_results <- dd_ML(
        brts,
        initparsopt = initparsopt,
        idparsopt = idparsopt,
        idparsfix = idparsfix,
        parsfix = parsfix,
        res = res,
        ddmodel = ddmodel,
        missnumspec = missnumspec,
        verbose = FALSE
        )    
    
    aic = AIC(dd_inf_results$loglik, 1) 
    
    #Save the results
    out_data <- data.frame(name = sub("\\.nwk$", "", tree),
                       likelihood = dd_inf_results$loglik,
                       AIC = aic,
                       estimated_lambda = dd_inf_results$lambda) 
    
    write.table(out_data,
    file = file.path(out_path, paste0("DDD_empiric_inference.csv")), 
    row.names = FALSE, 
    col.names = FALSE,  
    append = TRUE,
    quote=FALSE,
    sep=',')
       
}
