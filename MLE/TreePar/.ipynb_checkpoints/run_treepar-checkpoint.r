# -*- coding: utf-8 -*-
library(ape)
library(TreePar)
library(readr)
source("/workspace/phylo_estimation/MLE/TreePar/run_treepar_utils.r")
args = commandArgs(trailingOnly=TRUE)

# +
data_path <- as.character(args[1])
div_scenarios <- c("BD", "HE", "ME", "SAT", "SR", "WW")
n_tips <- as.integer(sub(".*/(\\d+)/?$", "\\1", data_path))
print(n_tips)
grid <- 0.2

out_path <-  paste0("/workspace/phylo_estimation/MLE/inference_data/", n_tips, "/")

# +
for (div in div_scenarios) {
    #Load data
    df <- read_delim(file.path(data_path, paste0(div, "_sim_", n_tips, "_TreePar.csv")), delim = "|")
    
    #Create empty CSV for saving results line by line 
    write.table(data.frame(index= numeric(),
                       #Real values for the erros 
                       real_a0 = numeric(),
                       real_a1 = numeric(),                       
                       real_r0 = numeric(),
                       real_r1 = numeric(),
                       real_t = numeric(),
                       real_frac_1 = numeric(),
                       
                       #MLE values for CBD
                       likelihood_cbd = numeric(),
                       aic_cbd = numeric(),
                       estimated_a = numeric(),
                       estimated_r = numeric(),
                       
                       
                       #MLE values for shift
                       likelihood_shift = numeric(),
                       aic_shift = numeric(),
                       estimated_a0 = numeric(),
                       estimated_a1 = numeric(),                       
                       estimated_r0 = numeric(),
                       estimated_r1 = numeric(),
                       estimated_t = numeric(),
                       
                       
                       #MLE values for me
                       likelihood_me = numeric(),
                       aic_me = numeric(),
                       estimated_a_me = numeric(),
                       estimated_r_me = numeric(),
                       estimated_frac_1_me = numeric(),  
                       estimated_time_me = numeric()
                       ),
                       
            file = file.path(out_path, paste0("TreePar_inference_", n_tips, "_", div, ".csv")),
            row.names = FALSE, 
            col.names = TRUE,
            quote=FALSE,
            sep=',')
    
    
    for (i in seq_len(nrow(df))) {
        #Load trees and data

        tree_data <- generate_tree_data(df$tree[i])
        rho <- df$frac0[i]
        
        #Perform inference
        time_inference <- bd.shifts.optim(tree_data[[1]], c(rho, 1), grid, tree_data[[2]], tree_data[[3]])
        extinction_inference <- bd.shifts.optim(tree_data[[1]], c(rho,1), grid, tree_data[[2]], tree_data[[3]], ME=TRUE)
        
        #Process shift results   

        time_inference_data <- process_time_inference(unlist(time_inference[[2]]))
        cbd_inference_data <- time_inference_data$cbd
        shift_inference_data <- time_inference_data$shift
        
        #Process ME results
        me_inference_data <- process_extinction_inference(unlist(extinction_inference[[2]]))
        
        
        #Save the results
        real_data_row <- df[i, ]
        csv_path <- file.path(out_path, paste0("TreePar_inference_", n_tips, "_", div, ".csv"))

        save_inference_results(
          csv_path = csv_path,
          idx = i,
          real_data_row = real_data_row,
          cbd_inference_data = cbd_inference_data,
          shift_inference_data = shift_inference_data,
          me_inference_data = me_inference_data
         )
    }

    
}

