#---------------------------------------------------------------
# Functions to fit models in adapt
#---------------------------------------------------------------

safe_glm <- function(formula, family, data, weights = NULL,
                      ...){
    options(warn = -1)

    formula <- as.formula(formula)
    if (family$link %in% c("inverse", "log")){
        fit <- try(glm(formula, family, data, weights, ...),
                   silent = TRUE)
        if (class(fit)[1] == "try-error"){
            mod_mat <- model.matrix(formula, data = data)
            p <- ncol(mod_mat) - 1
            start <- c(1, rep(0, p))
            fit <- glm(formula, family, data, weights,
                       start = start, ...)
        }
    } else {
        fit <- glm(formula, family, data, weights, ...)
    }

    fitv <- as.numeric(
        predict(fit, type = "response")
        )

    df <- fit$rank
    info <- list(df = df)

    options(warn = 0)
    # Return the model fit for the user to be able to access as well:
    return(list(fitv = fitv, info = info,
                model_fit = fit))
}

safe_gam <- function(formula, family, data, weights = NULL,
                      ...){
    options(warn = -1)

    formula <- as.formula(formula)
    if (family$link %in% c("inverse", "log")){
        fit <- try(mgcv::gam(formula, family, data, weights, ...),
                   silent = TRUE)
        if (class(fit)[1] == "try-error"){
            mod_mat <- model.matrix(formula, data = data)
            p <- ncol(mod_mat) - 1
            start <- c(1, rep(0, p))
            fit <- mgcv::gam(formula, family, data, weights,
                             start = start, ...)
        }
    } else {
        fit <- mgcv::gam(formula, family, data, weights, ...)
    }

    fitv <- as.numeric(
        predict(fit, type = "response")
        )

    df <- fit$rank
    info <- list(df = df)

    options(warn = 0)
    # Return the model fit for the user to be able to access as well:
    return(list(fitv = fitv, info = info,
                model_fit = fit))
}

safe_glmnet <- function(x, y, family, weights = NULL,
                        ...){
    options(warn = -1)

    if (class(family)[1] == "family"){
        family <- family$family
    }

    if (family %in% c("gaussian", "binomial", "poisson", "multinomial", "cox", "mgaussian")){
        if (is.null(weights)){
            fit <- glmnet::cv.glmnet(x, y,
                                     family = family, ...)
        } else {
            weights <- pminmax(weights, 1e-5, 1-1e-5)
            fit <- glmnet::cv.glmnet(x, y, weights,
                                     family = family, ...)
        }
    } else if (family == "Gamma"){
        if (is.null(weights)){
            fit <- HDtweedie::cv.HDtweedie(x, y, p = 2,
                                           standardize = TRUE,
                                           ...)
        } else {
            weights <- pminmax(weights, 1e-5, 1-1e-5)
            fit <- HDtweedie::cv.HDtweedie(x, y, p = 2,
                                           weights = weights,
                                           standardize = TRUE,
                                           ...)
        }
    }

    fitv <- as.numeric(
        predict(fit, newx = x, s = "lambda.min",
                type = "response")
        )

    beta <- coef(fit, s = "lambda.min")
    vi <- as.numeric(beta != 0)[-1]
    df <- sum(vi) + 1
    info <- list(df = df, vi = vi)

    options(warn = 0)
    # Return the model fit for the user to be able to access as well:
    return(list(fitv = fitv, info = info,
                model_fit = fit))
}

# Function for training xgboost model:
safe_xgboost <- function(x, y, family, weights = NULL, ...) {

  options(warn = -1)

  # For now we assume that the provided family argument is just a string
  # indicating what type of function to use for the xgboost. This will decide
  # what the objective argument is
  if (class(family)[1] == "family"){
    family <- family$family
  }

  # With the way xgboost is set up, each of the possible options will require
  # separate instances, can just use a named vector to access each:
  family_list <- c("gaussian" = "reg:linear",
                   "binomial" = "binary:logistic",
                   "poisson" = "count:poisson",
                   "multinomial" = "multi:softmax",
                   "cox" = "survival:cox",
                   "Gamma" = "reg:gamma")

  # If weights are provided then make sure they are within the bounds
  if (!is.null(weights)) {
    weights <- pminmax(weights, 1e-5, 1 - 1e-5)
  }

  # Fit the model using the provided family type as the objective with the
  # additional passed in arguments:
  fit <- xgboost::xgboost(data = x, label = y, weight = weights,
                          objective = family_list[[family]],
                          nrounds=5, # nrounds has no default value in xgboost
                          ...)

  # Get the predicted values using the boosting model
  fitv <- predict(fit, newdata = x)

  info <- "empty"

  options(warn = 0)
  # Return the model fit for the user to be able to access as well:
  return(list(fitv = fitv, info = info,
              model_fit = fit))
}

# Function for training bam model with options for running in parallel

safe_bam <- function(formula, family, data, weights = NULL,
                     ...){
  options(warn = -1)

  # Grab the additional arguments provided:
  model_args <- list(...)

  # Now check to see if the is_parallel argument is provided, check if it
  # set to be TRUE and then see if the number of clusters was provided.
  # This is assuming that the user provides the arguments `is_parallel` and
  # `n_clusters`:

  # First see if parallel is an argument:
  if (any(names(model_args) == "is_parallel")) {

    # See if it is TRUE and if there is more than one core found:
    if (model_args$is_parallel & (parallel::detectCores() > 1)) {

      # Now check to see if n_clusters was provided:
      if (any(names(model_args) == "n_clusters")) {

        # Now set the number of cores based on this value:
        if (parallel::detectCores() >= model_args$n_clusters) {
          cl <- parallel::makeCluster(model_args$n_clusters)

          # Remove n_clusters from the list
          model_args$n_clusters <- NULL

        } else {
          # Just use the number of cores:
          cl <- parallel::makeCluster(parallel::detectCores())
        }
      } else {
        # Again just use the actual number:
        cl <- parallel::makeCluster(parallel::detectCores())
      }


    }

    # Remove parallel from args:
    model_args$is_parallel <- NULL

    # Stop when the function exits:
    on.exit(parallel::stopCluster(cl))

  } else {
    # Otherwise there do not run in parallel:
    cl <- NULL
  }


  formula <- as.formula(formula)



  if (family$link %in% c("inverse", "log")){

    # Create a wrapper function for the then modified argument list to
    # be passed into easier that assumes there are the global variables
    # to be used - this is a little odd but allows for modification
    # to ... with the parallel arguments above and then use do.call
    bam_wrapper <- function(...) mgcv::bam(formula, family, data, weights, ...)

    fit <- try(do.call(bam_wrapper, model_args),
               silent = TRUE)
    if (class(fit)[1] == "try-error"){
      mod_mat <- model.matrix(formula, data = data)
      p <- ncol(mod_mat) - 1
      start <- c(1, rep(0, p))


      bam_wrapper <- function(...) mgcv::bam(formula, family, data, weights, start = start, ...)
      fit <- do.call(bam_wrapper, model_args)
    }
  } else {
    bam_wrapper <- function(...) mgcv::bam(formula, family, data, weights, ...)
    fit <- do.call(bam_wrapper, model_args)
  }

  fitv <- as.numeric(
    predict(fit, type = "response")
  )

  df <- fit$rank
  info <- list(df = df)

  options(warn = 0)
  # Return the model fit for the user to be able to access as well:
  return(list(fitv = fitv, info = info,
              model_fit = fit))
}
