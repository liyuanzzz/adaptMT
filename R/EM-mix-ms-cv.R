##------------------------------------------------------------------------------
# Model selection based on partially masked data using cross-validation
#-------------------------------------------------------------------------------

EM_mix_ms_cv <- function(x, pvals, s, dist, models,
                         # In addition to the above, the CV version needs the
                         # masking labels to use for the sampling:
                         mask,
                         # As well as the number of folds (default is 5):
                         n_folds = 5,
                         # Additionally provide the two predict-helpers functions
                         # that will be used for generating the holdhout parameters:
                         pred_model_pi, pred_model_mu, # NOTE: this assumes for now that all models are in the same family... could change later
                         params0 = list(pix = NULL, mux = NULL),
                         niter = 20, tol = 1e-4,
                         verbose = TRUE,
                         type = "unweighted",
                         masking_fun) {

  m <- length(models)
  if (verbose) {
    cat(paste0("Model selection with ", n_folds, "-fold cross-validation starts!\n"))
    cat("Shrink the set of candidate models or number of folds if it is too time-consuming.")
    cat("\n")
    pb <- txtProgressBar(min = 0, max = m, style = 3, width = 50)
    cat("\n")
  }

  # Generate a list of the hold-out indices - use the createFolds function in
  # the caret package to preserve the masking status (if there is enough otherwise
  # just randomly generated folds):
  k_cv_holdout_i <- caret::createFolds(mask, k = n_folds)

  # Now proceed to create for each of the models, the holdout parameter values
  # and resulting log-likelihood sums: (NOTE can replace this later with parallel version)
  model_results <- sapply(1:m,
                          function(model_i) {
                            model <- complete_model(models[[model_i]], dist)
                            # Next loop through the folds:
                            model_cv_loglik <- sapply(1:n_folds,
                                                       function(fold_i) {
                                                         # Set up the training data params if not null:
                                                         if (is.null(params0$pix) || is.null(params0$mux)) {
                                                           train_params0 <- list(pix = NULL, mux = NULL)
                                                         } else {
                                                           train_params0 <- list(pix = params0$pix[-k_cv_holdout_i[[fold_i]]],
                                                                                 mux = params0$mux[-k_cv_holdout_i[[fold_i]]])
                                                         }
                                                         fit <- try(
                                                           # Use as.matrix again to handle the case
                                                           # where only one covariate is used:
                                                           EM_mix(as.matrix(x[-k_cv_holdout_i[[fold_i]],]),
                                                                  pvals[-k_cv_holdout_i[[fold_i]]],
                                                                  s[-k_cv_holdout_i[[fold_i]]], dist,
                                                                  model, train_params0, niter, tol,
                                                                  type = type, masking_fun = masking_fun),
                                                           silent = TRUE
                                                         )
                                                         if (class(fit)[1] == "try-error"){
                                                           warning(paste0("Model ", model_i, "with fold ",
                                                                          fold_i," fails."))
                                                           NA
                                                         } else {
                                                           # Use the fitted parameters on the training
                                                           # data to compute the expected loglikelihood
                                                           # on the holdout p-values.

                                                           # First compute the values of pix and mux using
                                                           # the model fits (again use as.matrix for x to
                                                           # handle the situation with only 1 covariate):
                                                           test_pix <- pred_model_pi(fit$model_fit$pi,
                                                                                     as.matrix(x[k_cv_holdout_i[[fold_i]],]))
                                                           test_mux <- pred_model_mu(fit$model_fit$mu,
                                                                                     dist,
                                                                                     as.matrix(x[k_cv_holdout_i[[fold_i]],]))

                                                           # Get the E-step parameters:
                                                           test_Estep <- Estep_mix(pvals[k_cv_holdout_i[[fold_i]]],
                                                                                         s[k_cv_holdout_i[[fold_i]]],
                                                                                           dist, test_pix, test_mux,
                                                                                   masking_fun = masking_fun)

                                                           # Now return the holdout log-likelihood:
                                                           EM_loglik(pvals[k_cv_holdout_i[[fold_i]]], dist,
                                                                     test_pix, test_mux,
                                                                     test_Estep$Hhat, test_Estep$bhat, masking_fun)
                                                         }
                                                       })
                            # Return sum of holdout log-likelihood:
                            loglik_sum <- sum(unlist(model_cv_loglik), na.rm = TRUE)
                            #browser()
                            if (verbose) {
                              setTxtProgressBar(pb, model_i)
                              cat("\n")
                            }
                            return(loglik_sum)
                          })
  model_results <- unlist(model_results)
  #browser()

  if (all(is.na(model_results)) | all(is.infinite(model_results))) {
    stop("All models fail.")
  }

  # Otherwise which was the best model:
  best_model_i <- which.max(model_results)
  if (verbose) {
    cat(paste0("Selected model parameter choice: ", best_model_i, "!"))
    cat("\n")
  }
  # Now just fit the best model on all data:
  model <- complete_model(models[[best_model_i]], dist)
  final_fit <- EM_mix(x, pvals, s, dist, model, params0, niter, tol,
                      type = type, masking_fun = masking_fun)

  loglik <- final_fit$loglik
  params <- final_fit$params
  best_model <- models[[best_model_i]]
  best_model_info <- final_fit$info
  best_model_fit <- final_fit$model_fit

  return(list(model = best_model,
              params = params,
              info = best_model_info,
              model_fit = best_model_fit,
              # Also return the model holdout likelihood values to return:
              model_holdout_ll_sums = model_results))
}
