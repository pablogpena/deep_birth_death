# -*- coding: utf-8 -*-
library(TreeSim)

data <- data.frame(
  n_tips = numeric(),
  r0 = numeric(),
  r1 = numeric(),
  a0 = numeric(),
  a1 = numeric(),
  time = numeric(),
  frac0 = numeric(),
  frac1 = numeric(),
  tree = character(),
  stringsAsFactors = FALSE
)

# +
#Simulation conditions
n_sim <- 10000

mu <- c(0)
frac <- c(1)
#Centinels values
r1 <- 1
a0 <- 1
a1 <- 1
frac0 <- 1
frac1 <- frac0
times <- c(0)
# -

n_tips <- c(674, 489, 87)
for (n in n_tips) {
    no_fossil_path <- paste("/workspace/deep_birth_death/simulations/", n, "_10k/", sep = "")
    print(no_fossil_path)
    k <- as.integer(round(n + 1))
    for (i in 1:n_sim) {
    
        lambda <- runif(1, 0.01, 4)
        #k <- runif(1, n + 1, n*1.5)
        #k <- as.integer(round(as.numeric(k)))
        
        #Simulation
        simulation <- sim.rateshift.taxa(n, 1, lambda, mu, frac0, complete= FALSE, K = k)
    
        tree <- write.tree(simulation, append=TRUE, digits=10, tree.names=TRUE)[1]
        
        row <- c(n, lambda, r1, a0, a1, times, frac0, frac1, tree)
        ## Save .csv row
        data <- rbind(data, row)
        if(i%%1==0) {
            print(paste(Sys.time(), i, " DD simulations processed"))
             flush.console()
        }
    }
    
    file_path <- paste(no_fossil_path, "SAT_sim_no_fossil", n_sim, ".csv", sep = "")
    write.table(data, file = file_path, sep = "|", quote = FALSE, row.names = FALSE,
                col.names = c("n_tips", "r0", "r1", "a0", "a1", "time", "frac0", "frac1", "tree"))
}   


