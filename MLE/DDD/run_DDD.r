# -*- coding: utf-8 -*-
library(DDD)
library(ape)
library(readr)
args = commandArgs(trailingOnly=TRUE)

AIC <- function(Lik, k) {
    AIC <- 2*k -2*(Lik) #Lik here is positive, as DDD returns the loglikelihood in positive 
    return(AIC)
}

# +
data_path <- as.character(args[1])
div_scenarios <- c("BD", "HE", "ME", "SAT", "SR", "WW")
n_tips <- as.integer(sub(".*/(\\d+)/?$", "\\1", data_path))

k <- n_tips + 1

out_path <-  paste0("/workspace/deep_birth_death/MLE/inference_data/", n_tips, "/")

# +
for (div in div_scenarios) {
    #Load data
    df <- read_delim(file.path(data_path, paste0(div, "_sim_", n_tips, "_TreePar.csv")), delim = "|")
    
    #Create empty CSV for saving results line by line 
    write.table(
            data.frame(
            index = numeric(),
            real_lambda = numeric(), 
            likelihood = numeric(),
            AIC = numeric(),
            estimated_lambda = numeric()
            ),
                       
            file = file.path(out_path, paste0("DDD_inference_", n_tips, "_", div, ".csv")),
            row.names = FALSE, 
            col.names = TRUE,
            quote=FALSE,
            sep=',')
    
    
    for (i in seq_len(nrow(df))) {
        #Load trees and data

        tree <- read.tree(text = df$tree[i])
        brts <- ape::branching.times(tree)
    
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
        out_data <- data.frame(index=i,
                       real_lambda = df$r0[i],
                       likelihood = dd_inf_results$loglik,
                       AIC = aic,
                       estimated_lambda = dd_inf_results$lambda) 
        
        write.table(out_data,
            file = file.path(out_path, paste0("DDD_inference_", n_tips, "_", div, ".csv")), 
            row.names = FALSE, 
            col.names = FALSE,  
            append = TRUE,
            quote=FALSE,
            sep=',')
        
        
        csv_path <- file.path(out_path, paste0("TreePar_inference_", n_tips, "_", div, ".csv"))

    }

    
}
