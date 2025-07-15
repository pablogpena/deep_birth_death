# -*- coding: utf-8 -*-
generate_tree_data <- function(tree_nwk) {
  tr <- read.tree(text = tree_nwk)
  n_tips <- Ntip(tr)
  tr <- getx(tr)
  start <- tr[1]
  end <- tr[n_tips - 1]
  return(list(tr = tr, start = start, end = end))
}

process_time_inference <- function(inference_data) {
  
  #CBD data
  cbd_inference_data <- list(
    likelihood_cbd = inference_data[1],
    turnover_rate = inference_data[2],
    diversification_rate = inference_data[3]
  )
  
  #Shift data
  shift_inference_data <- list(
    likelihood_shift = inference_data[4],
    turnover_0_rate = inference_data[5],
    turnover_1_rate = inference_data[6],
    diversification_0_rate = inference_data[7],
    diversification_1_rate = inference_data[8],
    time = inference_data[9]
  )
  
  return(list(
    cbd = cbd_inference_data,
    shift = shift_inference_data
  ))
}

process_extinction_inference <- function(inference_data) {
  
  likelihood_me <- inference_data[4]
  turnover_rate_me <- inference_data[5]
  diversification_rate_me <- inference_data[6]
  magnitude_me <- inference_data[7]
  time <- inference_data[8]
  
  return(list(
    likelihood = likelihood_me,
    turnover_rate = turnover_rate_me,
    diversification_rate = diversification_rate_me,
    magnitude = magnitude_me,
    time = time
  ))
}

AIC <- function(Lik, k) {
    AIC <- 2*k -2*(-Lik)
    return(AIC)
}

save_inference_results <- function(csv_path, idx,
                                   real_data_row,
                                   cbd_inference_data,
                                   shift_inference_data,
                                   me_inference_data) {

  out_data <- data.frame(
    index = idx,

    # Real values (extraídos desde la fila real_data_row)
    real_a0 = real_data_row$a0,
    real_a1 = real_data_row$a1,
    real_r0 = real_data_row$r0,
    real_r1 = real_data_row$r1,
    real_t = real_data_row$time,
    real_frac_1 = real_data_row$frac1,

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

  # Guardamos la fila en el CSV
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
