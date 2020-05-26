import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter


mcrlib = importr("mcr")
kernlablib = importr("kernlab")

class SVMMCR(object):

    def __init__(self):
        self.cv_sigma = 1.857097
        self.cv_alpha = 0.002424462
        self.cv_min_loss = 2.808806

    def disable_R_warnings(self):
        r("options(warn=-1)")
    
    def enable_R_warnings(self):
        r("options(warn=0)")

    def learn_reference_model_params(self, X, y):
        # Due to computational complexity we are not going to try a grid search
        # Rather we will simply find the best sigma, then using the best sigma
        # find the best alpha. This is per: https://github.com/aaronjfisher/mcr-supplement/blob/master/propublica-analysis/propublica_kernel.md

        with localconverter(ro.default_converter + pandas2ri.converter):
            r_X = ro.conversion.py2rpy(X) 
        with localconverter(ro.default_converter + numpy2ri.converter):
            r_X = ro.conversion.py2rpy(X.values) 
            r_y = ro.conversion.py2rpy(y) 
            r_X_cols = ro.conversion.py2rpy(X.columns.values)

        r.assign('X', r_X)
        r.assign('X_cols',list(X.columns.values))
        r("colnames(X) <- X_cols")
        r.assign('y', r_y)


        rcode = """
        # tr denotes "training" data
        mu_tr <- mean(y)

        p <- dim(X)[2]
        len_s <- 40

        sigma_seq <- p^seq(-5,5,length=len_s)
        cv_err_sigma_regression <- rep(NA,len_s)
        pb <- txtProgressBar(min = 1, max = len_s, char = "=", 
                style = 3)
        for(i in 1:len_s){
            try({ # may be singular, in which case sigma is too small
                cv_err_sigma_regression[i] <- CV_kernel(y=y,X=X, type='regression',
                    kern_fun=rbfdot(sigma_seq[i]),
                    dat_ref=NA, n_folds=5, warn_internal=FALSE)
            })
            setTxtProgressBar(pb, i)
        }

        sigma_regression <- sigma_seq[which(cv_err_sigma_regression==min(cv_err_sigma_regression, na.rm=TRUE))]

        #When bandwidth is too small, some test points may not be close to *any* reference point,
        #which can lead to NaNs and zeros in the kernel regression.
        #This is why some elements of cv_err_sigma_regression are NaN

        kern_fun <- rbfdot(sigma_regression)

        # Note, some rows of X are identical.
        # These will be automatically dropped from 
        # reference matrices
        mean(duplicated(X))

        len_a <- 40
        alpha_seq <- 10^seq(-4,2,length=len_a)
        cv_KLS <- rep(NA,len_a)
        pb <- txtProgressBar(min = 1, max = len_a, char = "=", 
                style = 3)

        for(i in 1:len_a){
            try({
                cv_KLS[i] <- CV_kernel(y=y-mu_tr,X=X,alpha=alpha_seq[i], type='RKHS',
                kern_fun=kern_fun, dat_ref=NA, n_folds=10,
                warn_internal=FALSE, warn_psd=FALSE, warn_duplicate = FALSE)
            })
            setTxtProgressBar(pb, i)
        }

        min_cv_loss <- min(cv_KLS,na.rm=TRUE)
        alpha_cv <- alpha_seq[which(cv_KLS==min_cv_loss)[1]]
        """
        r(rcode)

        with localconverter(ro.default_converter + numpy2ri.converter):
            self.cv_min_loss = float(r("min_cv_loss")[0])
            self.cv_alpha = float(r("alpha_cv")[0])
            self.cv_sigma = float(r("sigma_regression")[0])
        print('+++++++++++++++++++++++++++++++++++')
        print('cv loss: {} alpha cv: {} sigma: {}'.format(self.cv_min_loss, self.cv_alpha, self.cv_sigma))
    

    def fit_reference_model(self, X, y):
        # Fits the reference model
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_X = ro.conversion.py2rpy(X) 
        with localconverter(ro.default_converter + numpy2ri.converter):
            r_X = ro.conversion.py2rpy(X.values) 
            r_y = ro.conversion.py2rpy(y) 
            r_X_cols = ro.conversion.py2rpy(X.columns.values)

        r.assign('X', r_X)
        r.assign('X_cols',list(X.columns.values))
        r("colnames(X) <- X_cols")
        r.assign('y', r_y)

        sigma = self.cv_sigma
        alpha = self.cv_alpha

        rcode = """

        sigma_regression = {0:.20f}
        alpha_cv = {1:.20f}

        kern_fun <- rbfdot(sigma_regression)
        mean(duplicated(X))
        mu_tr <- mean(y)


        K_D <- as.matrix(kernelMatrix(x=X,kernel=kern_fun))

        eK_D <- eigen(K_D)


        ###### Train f_s reference model
        X_ref <- X[!duplicated(X),]
        y_ref <- y[!duplicated(X)]

        ssts_tr <- get_suff_stats_kernel( y=y-mu_tr, X=X,kern_fun=kern_fun,dat_ref=X_ref)
        w_ref <- fit_lm_regularized(suff_stats =ssts_tr, tol = 10^-9, alpha = alpha_cv)

        (r_constraint <- norm_RKHS(model=w_ref, K_D=ssts_tr$reg_matrix))
        """.format(sigma, alpha)
        r(rcode)


    def predict(self, X, y):
        # Predict using the reference model
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_X = ro.conversion.py2rpy(X) 
        with localconverter(ro.default_converter + numpy2ri.converter):
            r_X = ro.conversion.py2rpy(X.values) 
            r_y = ro.conversion.py2rpy(y) 
            r_X_cols = ro.conversion.py2rpy(X.columns.values)

        r.assign('X', r_X)
        r.assign('X_cols',list(X.columns.values))
        r("colnames(X) <- X_cols")
        r.assign('y', r_y)
        # rcode = """
        # pred_RKHS <- function(X=X, kern_fun=kern_fun, w_ref)
        # """
        rcode = """
        kernel_regression_prediction <- function(X, X_ref, y_ref, kern_fun){
            #Simple kernel smoother estimator
            K <- as.matrix(kernelMatrix(x=X,y=X_ref, kernel=kern_fun))
            W <- K / tcrossprod(rowSums(K), rep(1,length(y_ref)))
            c(W %*% y_ref)

            # !! Note - See KRLS package to see how they handle binary covariates
                # However, this package does not appear to let you specify a present reference dataset; it always uses all your data.

        }

        preds <- kernel_regression_prediction(X=X, X_ref=X_ref, y_ref=y_ref, kern_fun=kern_fun)
        """
        r(rcode)

        with localconverter(ro.default_converter + numpy2ri.converter):
            preds = ro.conversion.rpy2py(r("preds")) 

        return preds

    def get_mcr(self, X, y, vars2permute):
        # Performs MCR

        with localconverter(ro.default_converter + pandas2ri.converter):
            r_X = ro.conversion.py2rpy(X) 
        with localconverter(ro.default_converter + numpy2ri.converter):
            r_X = ro.conversion.py2rpy(X.values) 
            r_y = ro.conversion.py2rpy(y) 
            r_X_cols = ro.conversion.py2rpy(X.columns.values)

        r.assign('X', r_X)
        r.assign('X_cols',list(X.columns.values))
        r("colnames(X) <- X_cols")
        r.assign('y', r_y)
     
        min_cv_loss = self.cv_min_loss

        
        rcode = """
            st = system.time({{

            p1_sets = list("permuted"= as.integer(c({0})))

            min_cv_loss = {1:.20f} # value from CV
            eps_multiplier <- 0.1

            # te denotes "test" data rather than "train" (tr)
            te_kernel_precomputed <- lapply(p1_sets, function(set){{
                precompute_mcr_objects_and_functions(
                    y=y-mu_tr, X=X,
                    p1=set,
                    model_class_loss='kernel_mse',
                    loop_ind_args = list(
                        reg_threshold=r_constraint,
                        kern_fun = kern_fun,
                        dat_ref=X_ref,
                        nrep_sample=2,
                        tol = 10^-8,
                        verbose=TRUE,
                        warn_psd=TRUE,
                        warn_duplicate = TRUE,
                        warn_dropped = TRUE)
                    )
            }})

            MR_ref_te <- lapply(te_kernel_precomputed, function(pc)
                get_MR_general(model=w_ref,
                    precomputed = pc
                ))
            str(MR_ref_te) #tag-MR-ref-TE

            (loss_ref_te <- get_e0_lm(model = w_ref, suff_stats = te_kernel_precomputed[[1]]$suff_stats))
            (eps_ref_te <- c(loss_ref_te + eps_multiplier * min_cv_loss))
            # tag-w_S-held-out-Err

            mcr_te <- lapply(te_kernel_precomputed, function(pc) 
                    get_empirical_MCR(eps=eps_ref_te, precomputed = pc, tol_mcr=2^-10)
                    )

            }})
            """.format(','.join([str(x+1) for x in vars2permute]), min_cv_loss) # +1 due to R indexing
        r(rcode)


        eps_ref_ts = np.asarray(r("eps_ref_te"))
        mcr = np.asarray(r("mcr_te$permuted$range")) 

        return mcr,eps_ref_ts