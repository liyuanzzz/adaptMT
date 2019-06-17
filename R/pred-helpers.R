#---------------------------------------------------------------
# Helpers for predicting to use in the cross-validation
#---------------------------------------------------------------

# Define the XGBoost predict functions:

pred_xgboost_pi <- function(fit, new_data) {
  pix <- predict(fit, newdata = new_data)
  pminmax(pix, 0, 1)
}

pred_xgboost_mu <- function(fit, dist, new_data) {
  mux <- predict(fit, newdata = new_data)
  if (dist$family$family == "Gamma"){
    mux <- pmax(mux, 1)
  } else if (dist$family$family == "gaussian"){
    mux <- pmax(mux, 0)
  }
  return(mux)
}
