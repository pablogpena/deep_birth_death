# -*- coding: utf-8 -*-
library(TreePar)
library(ape)
source("/workspace/deep_birth_death/src/MLE_utils/run_treepar_utils.r")

empiric_path <- "/workspace/deep_birth_death/empirical/"
tree_names <- c("eucalypts.nwk", "conifers.nwk", "cetaceans.nwk")
out_path <- "/workspace/deep_birth_death//MLE/inference_data/empiric"

#Create empty CSV for saving results line by line 
write.table(data.frame(name = character(),
               
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
            
            file = file.path(out_path, paste0("TreePar_empiric_inference.csv")),
            row.names = FALSE, 
            col.names = TRUE,
            quote=FALSE,
                sep=',') 

# +
for (tree in tree_names) {

    
    #Load and encode trees
    tr <- read.tree(paste0(empiric_path, tree))
    n_tips <- Ntip(tr)
    tr <- getx(tr)
    
    #Set parameters for MLE 
    start <- tr[1]
    end <- tr[n_tips - 1]
    rho <- 1
    grid <- 0.2
    
    #Perform inference
    time_inference <- bd.shifts.optim(tr, c(rho, 1), grid, start, end)
    extinction_inference <- bd.shifts.optim(tr, c(rho,1), grid, start, end, ME=TRUE)
    
    
    
    #Process shift results
    time_inference_data <- process_time_inference(unlist(time_inference[[2]]))
    cbd_inference_data <- time_inference_data$cbd
    shift_inference_data <- time_inference_data$shift    
    
    #Process ME results 
    me_inference_data <- process_extinction_inference(unlist(extinction_inference[[2]]))    
    
    #Save the results 
    csv_path <- file.path(out_path, paste0("TreePar_empiric_inference.csv"))
    
    out_data <- data.frame(
        name = sub("\\.nwk$", "", tree),
    
        # CBD inference
        likelihood_cbd = cbd_inference_data$likelihood_cbd,
        aic_cbd = AIC(cbd_inference_data$likelihood_cbd, 2),
        estimated_a = cbd_inference_data$turnover_rate,
        estimated_r = cbd_inference_data$diversification_rate,
    
        # Shift inference
        likelihood_shift = shift_inference_data$likelihood_shift,
        aic_shift = AIC(shift_inference_data$likelihood_shift, 5),
        estimated_a0 = shift_inference_data$turnover_0_rate,
        estimated_a1 = shift_inference_data$turnover_1_rate,
        estimated_r0 = shift_inference_data$diversification_0_rate,
        estimated_r1 = shift_inference_data$diversification_1_rate,
        estimated_t = shift_inference_data$time,
    
        # ME inference
        likelihood_me = me_inference_data$likelihood,
        aic_me = AIC(me_inference_data$likelihood, 4),
        estimated_a_me = me_inference_data$turnover_rate,
        estimated_r_me = me_inference_data$diversification_rate,
        estimated_frac_1_me = me_inference_data$magnitude,
        estimated_time_me = me_inference_data$time
  )
       # Save row at CSV
       write.table(
           out_data,
           file = csv_path,
           row.names = FALSE,
           col.names = FALSE,
           append = TRUE,
           quote = FALSE,
           sep = ','
      )   
    
}
