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
                          ...)

  # Get the predicted values using the boosting model
  fitv <- predict(fit, newdata = x)

  info <- "empty"

  options(warn = 0)
  # Return the model fit for the user to be able to access as well:
  return(list(fitv = fitv, info = info,
              model_fit = fit))
}

