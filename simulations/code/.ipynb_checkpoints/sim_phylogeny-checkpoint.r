# -*- coding: utf-8 -*-
# +
# Simulating birth dead constant
args = commandArgs(trailingOnly=TRUE)
library(TreeSim)
source("/workspace/phylo_estimation/simulations/code/Stadler-LTTplots-functions.R")
source("/workspace/phylo_estimation/simulations/config_sim.r")

data <- data.frame(n_tips=numeric(), r0=numeric(), r1=numeric(), a0=numeric(), a1=numeric(), time=numeric(),
                   frac0=numeric(), frac1=numeric(), tree=character(), stringsAsFactors = FALSE)

i <- 0
while (i < n_sim) {
    r0 <- get_r0()
    a0 <- get_a0()
    frac0 <- get_frac0()
    
    r1 <- get_r1(r0)
    a1 <- get_a1(a0)
    frac1 <- get_frac1(frac0)
    time <- get_time()
    
    # Calculate mu and lambda
    mu0 <- (a0*r0)/(1-a0)
    mu1 <- (a1*r1)/(1-a1)
    lamb0 <- r0+mu0
    lamb1 <- r1+mu1
    
    # Simulate phylogenetic trees
    if(args[1] == "BD" | args[1] == "HE") {
        simulation <- sim.bd.taxa(n_tips, 1, lamb0, mu0, frac0, complete=complete)
    }
    else if(args[1] == "ME" |args[1] == "ME_PGP" | args[1] == "SR" | args[1] == "ME_rho" | args[1] == "ME_rho_PGP" | args[1] == "WW") {
        simulation <- sim.rateshift.taxa(n_tips, 1, c(lamb0, lamb1), c(mu0, mu1),
                                         c(frac0, frac1), c(0, time), complete=complete)
    }
    
    crown_age_matrix <- origin(simulation)
    df_crown_age <- as.data.frame(crown_age_matrix)
    crown_age <- df_crown_age$origintime
    crown_limit <- -time - time_limit

    if (length(df_crown_age$origintime) == 0){
        #print(paste(Sys.time(), " simulation failed, it contains less tips"))
        #flush.console()
    } else if (count_living(simulation[[1]]) != n_tips) {
        #print(paste(Sys.time(), " simulation failed, it contains more living tips"))
        #flush.console()
    } else if (count_dead(simulation[[1]]) != 0) {
        #print(paste(Sys.time(), " simulation failed, it contains more death tips"))
        #flush.console()
    } else if (crown_age >= crown_limit) {
        #print(paste(Sys.time(), " crown age lower than shift"))
        #flush.console()
    } else {
        # Save .csv row
        tree <- write.tree(simulation, append=TRUE, digits=10, tree.names=TRUE)[1]
        row <- c(n_tips, r0, r1, a0, a1, time, frac0, frac1, tree)
        data <- rbind(data, row)

        i <- i + 1

        if(i%%100==0) {
            print(paste(Sys.time(), i, args[1], " simulations processed"))
        }
    }
}

if (complete == TRUE) {
   file_path <- paste(raw_path, args[1], "_sim_", n_sim, ".csv", sep = "")
} else {
   file_path <- paste(no_fossil_path, args[1], "_sim_no_fossil", n_sim, ".csv", sep = "")
}

write.table(data, file = file_path, sep = "|", quote = FALSE, row.names = FALSE,
            col.names = c("n_tips", "r0", "r1", "a0", "a1", "time", "frac0", "frac1", "tree"))
