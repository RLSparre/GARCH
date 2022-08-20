import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm, t, chi2
from scipy.special import gamma as G
from tabulate import tabulate

class garch_models:
    
    class tests:
        
        def __init__(self, Y):
            self.Y = Y
            self.T = len(Y)
            
        def jarque_bera(self, significance_level = 0.05):
    
            normal_bool = True
            smpl_mean = np.mean(self.Y)

            smpl_skew = (1/self.T*sum((self.Y-smpl_mean)**3))/((1/self.T*sum((self.Y-smpl_mean)**2))**(3/2))
    
            smpl_kurt = (1/self.T*sum((self.Y-smpl_mean)**4))/((1/self.T*sum((self.Y-smpl_mean)**2))**2)
            smpl_exc_kurt = smpl_kurt - 3
    
            jb_ts = self.T/6 * (smpl_skew**2 + 0.25*(smpl_kurt - 3)**2)
    
            if self.T < 2000:
                print('The number of observations is less than 2000, and thus the distribution of the test statistic may not be well-behaved.')
    
            if jb_ts > chi2.ppf(1-significance_level, 2):
                print('The null of normality is rejected at a 5% significance level')
                normal_bool = False
            else:
                print('The null of normality is NOT rejected at a 5% significance level')
            
            return normal_bool, jb_ts

    class garch:
        
        def __init__(self, Y, dist='normal'):
            self.Y = Y
            self.T = len(Y)
            self.omega = None
            self.alpha = None
            self.beta = None
            self.nu = None
            self.hess = False
            self.se = None
            self.dist = dist
            
        def filter(self, omega, alpha, beta, nu=None):    
            sig2 = np.zeros(self.T)
            
            # initiate
            sig2[0] = self.T/(self.T-1)*np.var(self.Y)
            
            # compute variance
            for t in range(1, self.T):
                
                sig2[t] = omega + alpha*self.Y[t-1]**2 + beta*sig2[t-1]
            
            # calculate standardized errors
            z = self.Y/np.sqrt(sig2)
        
            # calculate LL
            if self.dist == 'normal':
                loglik = (- 0.5*self.T*np.log(2*np.pi) - 0.5*sum(np.log(sig2)) - 0.5 * sum(self.Y**2/sig2)) 
            elif self.dist == 'student_t':
                term1 = self.T*np.log(G((nu+1)/2)/(np.sqrt(np.pi*(nu-2))*G(nu/2)))
                term2 = 0.5*sum(np.log(sig2))
                term3 = (nu+1)/2*sum(np.log(1+(np.sqrt(sig2)*z)**2/(sig2*(nu-2))))
                loglik = term1 - term2 - term3
                
                
            res = [sig2, z, loglik]
            
            return res
        
        def obj_func(self, param):
            if self.dist == 'normal':
                omega, alpha, beta = param
                # check parameter conditions
                if sum(np.isfinite(param)) != len(param):
                    return 1e9
                if (omega < 0) | (alpha < 0) | (beta < 0):
                    return 1e9
                if alpha + beta >= 1:
                    return 1e9
                
                #retutn negative loglikelihood
                neg_loglik = -1*self.filter(omega, alpha, beta)[2]
                
            elif self.dist == 'student_t':
                omega, alpha, beta, nu = param
                # check parameter conditions
                if sum(np.isfinite(param)) != len(param):
                    return 1e9
                if (omega < 0) | (alpha < 0) | (beta < 0):
                    return 1e9
                if alpha + beta >= 1:
                    return 1e9
                if nu <2:
                    return 1e9
                # retutn negative loglikelihood
                neg_loglik = -1*self.filter(omega, alpha, beta, nu)[2]
                    
        
            return neg_loglik
        
        def mle(self):
            if self.dist == 'normal':
                if self.omega != None:
                    print('Parameters already optimized: \n')
                    print('omega = %.4f, alpha = %.4f, beta = %.4f' % (self.omega, self.alpha, self.beta))
                    return None
                
                res = minimize(self.obj_func, (0.1, 0.05, 0.9),
                            bounds=[(1e-5,None), (1e-5,None), (1e-5,None)], method='L-BFGS-B')
                
                opt_param = res.x
                self.omega, self.alpha, self.beta = opt_param
            
            elif self.dist == 'student_t':
                if self.omega != None:
                    print('Parameters already optimized: \n')
                    print('omega = %.4f, alpha = %.4f, beta = %.4f, nu = %.4f' % (self.omega, self.alpha, self.beta, self.nu))
                    return None
                    
                res = minimize(self.obj_func, (0.1, 0.05, 0.9, 8),
                    bounds=[(1e-5,None), (1e-5,None), (1e-5,None), (2.0001, None)], method='L-BFGS-B')
        
                opt_param = res.x
                self.omega, self.alpha, self.beta, self.nu = opt_param
                
            return res
        
        
        
        def std_err(self, h = 1e-4):
            '''
            Function that compute the standard errors using the asymptotic variance covariance matrix of the maximum
            likelihood estimator based on the Hessian og the log-likelihood, where the Hessian is evaluated at $\hat{\theta}$.
            The Hessian is approximated numerically using finite differences. Let $\varepsilon$ be defined as $\hat{\theta}$, where 
            $\hat{\theta}$ is the maximum likelihood estimator and $h$ is the numerical step size. Let $D$ be a $3 \times 3$ diagonal
            matrix, whose diagonal is equal to $\varepsilon$. Let $\varepsilon_{i}$ denote a $3$ dimensional vector with $1$ in 
            position $i$ and zero otherwise. Then, the $(i,j)$ entry of $H_{t}(\hat{\theta})$ is approximated as
            $$
            .....
            $$
            
            '''
            # check if parameters already optimized
            if self.omega == None:
                self.mle()
                
            # check if standard errors already calculated
            if self.hess != False:
                if self.dist == 'normal':
                    print('Standard errors already computed: \n')
                    print('omega_se = %.6f, alpha_se = %.6f, beta_se = %.6f' % (self.se[0], self.se[1], self.se[2]))
                elif self.dist == 'student_t':
                    print('Standard errors already computed: \n')
                    print('omega_se = %.6f, alpha_se = %.6f, beta_se = %.6f, nu_se = %.6f' % (self.se[0], self.se[1], self.se[2], self.se[3]))
                return None
            
            if self.dist == 'normal':
                param = np.array([self.omega, self.alpha, self.beta])
            elif self.dist == 'student_t':
                param = np.array([self.omega, self.alpha, self.beta, self.nu])
                
            eps = param * h
            eps_mat = np.diag(eps)
            i_mat = np.diag(np.repeat(1,len(param)))
            hess = np.zeros([len(param),len(param)])
            
            for i in range(len(param)):
                for j in range(len(param)):
                    
                    add_term_1 = np.matmul(eps_mat, i_mat[i])
                    add_term_2 = np.matmul(eps_mat, i_mat[j])
                                            
                    new_param = param + add_term_1 + add_term_2
                    l1 = -1*self.obj_func(new_param)
                    new_param = param+ add_term_1 - add_term_2
                    l2 = -1*self.obj_func(new_param)
                    new_param = param - add_term_1 + add_term_2
                    l3 = -1*self.obj_func(new_param)
                    new_param = param - add_term_1 - add_term_2
                    l4 = -1*self.obj_func(new_param)
                    
                    hess[i][j] = (l1-l2-l3+l4) / (4*eps[i]*eps[j])
            
            hat_vcv = np.linalg.inv(-1*hess)
            se =  np.sqrt(np.diag(hat_vcv))
            self.hess = True
            self.se = se
            
            return se
        
        def coef(self):
            if self.omega == None:
                self.mle()
            
            if self.hess == False:
                self.std_err()
            
            if self.dist == 'normal':
                param = np.array([self.omega, self.alpha, self.beta])
                row_titles = ['omega', 'alpha', 'beta']
            if self.dist == 'student_t':
                param = np.array([self.omega, self.alpha, self.beta, self.nu])
                row_titles = ['omega', 'alpha', 'beta', 'nu']
                
            t_val = param/self.se
            p_val = 2*(1-norm.cdf(np.abs(t_val)))
            
            col_titles = ['', 'Estimate', 'Std. Error', 't-value', 'Pr(>|t|)', '95% CI']

            
            aux_list = []
            for i in range(len(row_titles)):
                lower = param[i]-norm.ppf(0.975)*self.se[i]/np.sqrt(self.T)
                upper = param[i]+norm.ppf(0.975)*self.se[i]/np.sqrt(self.T)
                aux_list.append([row_titles[i], round(param[i],4), round(self.se[i],4), round(t_val[i],4), p_val[i], '[%.4f, %.4f]' % (lower, upper)])
                
            print(tabulate(aux_list, headers=col_titles))
            
            if self.dist == 'normal':
                print('Loglikelihood %.2f' % self.filter(self.omega, self.alpha, self.beta)[2])
            if self.dist == 'student_t':
                print('Loglikelihood %.2f' % self.filter(self.omega, self.alpha, self.beta, self.nu)[2])
            
        
        def fit(self):
            
            # check if parameters already optimized
            if self.omega == None:
                self.mle()
            
            # run garch filter with optimized parameters
            if self.dist == 'normal':
                return self.filter(self.omega, self.alpha, self.beta)
            if self.dist == 'student_t':
                return self.filter(self.omega, self.alpha, self.beta, self.nu)
            
            
        def plot(self):
            
            # call fit
            fit = self.fit()
            
            # initiate figure
            fig, ax = plt.subplots(3, 1, figsize=(20,12))
            axs = ax.ravel()
            
            # plot 1-step ahead conditional variance
            ax[0].plot(self.Y.index, fit[0])
            
            # plot ret and 95% CI
            ax[1].scatter(self.Y.index, self.Y)
            if self.dist == 'normal':
                ax[1].plot(self.Y.index, norm.ppf(0.975)*np.sqrt(fit[0]), c='red')
                ax[1].plot(self.Y.index, -norm.ppf(0.975)*np.sqrt(fit[0]), c='red')
                
                # qqplot
                sm.qqplot(fit[1], line='45', ax=ax[2])
                
            if self.dist == 'student_t':
                ax[1].plot(self.Y.index, t.ppf(0.975, self.nu)*np.sqrt(fit[0]), c='red')
                ax[1].plot(self.Y.index, -t.ppf(0.975, self.nu)*np.sqrt(fit[0]), c='red')
                
                # qqplot
                sm.qqplot(fit[1], t, distargs=(self.nu,), line='45', ax=ax[2])
            
            
            return None
        
    class gjr_garch:
        
        def __init__(self, Y, dist='normal'):
            self.Y = Y
            self.T = len(Y)
            self.omega = None
            self.alpha = None
            self.beta = None
            self.gamma = None
            self.nu = None
            self.hess = False
            self.se = None
            self.dist = dist
            
        def filter(self, omega, alpha, beta, gamma, nu = None):    
            
            
            sig2 = np.zeros(self.T)
            
            I = self.Y < 0
            # initiate
            sig2[0] = self.T/(self.T-1)*np.var(self.Y)
            
            # compute variance
            for t in range(1, self.T):
                
                sig2[t] = omega + (alpha + I[t-1]*gamma)*self.Y[t-1]**2 + beta*sig2[t-1]
            
            # calculate standardized errors
            z = self.Y/np.sqrt(sig2)
        
             # calculate LL
            if self.dist == 'normal':
                loglik = (- 0.5*self.T*np.log(2*np.pi) - 0.5*sum(np.log(sig2)) - 0.5 * sum(self.Y**2/sig2)) 
            elif self.dist == 'student_t':
                term1 = self.T*np.log(G((nu+1)/2)/(np.sqrt(np.pi*(nu-2))*G(nu/2)))
                term2 = 0.5*sum(np.log(sig2))
                term3 = (nu+1)/2*sum(np.log(1+(np.sqrt(sig2)*z)**2/(sig2*(nu-2))))
                loglik = term1 - term2 - term3
            
            res = [sig2, z, loglik]
            
            return res
        
        def obj_func(self, param):
            
            if self.dist == 'normal':
                omega, alpha, beta, gamma = param
                # check parameter conditions
                if sum(np.isfinite(param)) != len(param):
                    return 1e9
                if (omega < 0) | (alpha < 0) | (beta < 0):
                    return 1e9
                if alpha + gamma/2 + beta >= 1:
                    return 1e9
                
                #retutn negative loglikelihood
                neg_loglik = -1*self.filter(omega, alpha, beta, gamma)[2]
                
            elif self.dist == 'student_t':
                omega, alpha, beta, gamma, nu = param
                # check parameter conditions
                if sum(np.isfinite(param)) != len(param):
                    return 1e9
                if (omega < 0) | (alpha < 0) | (beta < 0):
                    return 1e9
                if alpha + gamma/2 + beta >= 1:
                    return 1e9
                if nu <2:
                    return 1e9
                # retutn negative loglikelihood
                neg_loglik = -1*self.filter(omega, alpha, beta, gamma, nu)[2]
        
            return neg_loglik
        
        def mle(self):
            if self.dist == 'normal':
                if self.omega != None:
                    print('Parameters already optimized: \n')
                    print('omega = %.4f, alpha = %.4f, beta = %.4f, gamma = %.4f' % (self.omega, self.alpha, self.beta, self.gamma))
                    return None
                
                res = minimize(self.obj_func, (0.1, 0.05, 0.9, 0.05),
                            bounds=[(1e-5,None), (1e-5,None), (1e-5,None), (1e-5,None)], method='L-BFGS-B')
                
                opt_param = res.x
                self.omega, self.alpha, self.beta, self.gamma = opt_param
                
            if self.dist == 'student_t':
                if self.omega != None:
                    print('Parameters already optimized: \n')
                    print('omega = %.4f, alpha = %.4f, beta = %.4f, gamma = %.4f, nu = %.4f' % (self.omega, self.alpha, self.beta, self.gamma, self.nu))
                    return None
                

                res = minimize(self.obj_func, (0.1, 0.05, 0.9, 0.05, 8),
                            bounds=[(1e-5,None), (1e-5,None), (1e-5,None), (1e-5,None), (2.0001, None)], method='L-BFGS-B')
                
                opt_param = res.x
                self.omega, self.alpha, self.beta, self.gamma, self.nu = opt_param
            
            
            return res
        
        
        
        def std_err(self, h = 1e-4):
            '''
            Function that compute the standard errors using the asymptotic variance covariance matrix of the maximum
            likelihood estimator based on the Hessian og the log-likelihood, where the Hessian is evaluated at $\hat{\theta}$.
            The Hessian is approximated numerically using finite differences. Let $\varepsilon$ be defined as $\hat{\theta}$, where 
            $\hat{\theta}$ is the maximum likelihood estimator and $h$ is the numerical step size. Let $D$ be a $3 \times 3$ diagonal
            matrix, whose diagonal is equal to $\varepsilon$. Let $\varepsilon_{i}$ denote a $3$ dimensional vector with $1$ in 
            position $i$ and zero otherwise. Then, the $(i,j)$ entry of $H_{t}(\hat{\theta})$ is approximated as
            $$
            .....
            $$
            
            '''
 
            # check if standard errors already calculated
            if self.hess != False:
                print('Standard errors already computed: \n')
                print('omega_se = %.6f, alpha_se = %.6f, beta_se = %.6f, gamma_se = %.6f' % (self.se[0], self.se[1], self.se[2], self.se[3]))
                return None
                if self.dist == 'normal':
                    print('Standard errors already computed: \n')
                    print('omega_se = %.6f, alpha_se = %.6f, beta_se = %.6f, gamma_se = %.6f' % (self.se[0], self.se[1], self.se[2], self.se[3]))
                elif self.dist == 'student_t':
                    print('Standard errors already computed: \n')
                    print('omega_se = %.6f, alpha_se = %.6f, beta_se = %.6f, gamma_se = %.6f, nu_se = %.6f'
                          % (self.se[0], self.se[1], self.se[2], self.se[3], self.se[4]))
                return None
            
            if self.dist == 'normal':
                param = np.array([self.omega, self.alpha, self.beta, self.gamma])
            elif self.dist == 'student_t':
                param = np.array([self.omega, self.alpha, self.beta, self.gamma, self.nu])
                
            eps = param * h
            eps_mat = np.diag(eps)
            i_mat = np.diag(np.repeat(1,len(param)))
            hess = np.zeros([len(param),len(param)])
            
            for i in range(len(param)):
                for j in range(len(param)):
                    
                    add_term_1 = np.matmul(eps_mat, i_mat[i])
                    add_term_2 = np.matmul(eps_mat, i_mat[j])
                                            
                    new_param = param + add_term_1 + add_term_2
                    l1 = -1*self.obj_func(new_param)
                    new_param = param+ add_term_1 - add_term_2
                    l2 = -1*self.obj_func(new_param)
                    new_param = param - add_term_1 + add_term_2
                    l3 = -1*self.obj_func(new_param)
                    new_param = param - add_term_1 - add_term_2
                    l4 = -1*self.obj_func(new_param)
                    
                    hess[i][j] = (l1-l2-l3+l4) / (4*eps[i]*eps[j])
            
            hat_vcv = np.linalg.inv(-1*hess)
            se =  np.sqrt(np.diag(hat_vcv))
            self.hess = True
            self.se = se
            
            return se
        
        def coef(self):
            if self.omega == None:
                self.mle()
            
            if self.hess == False:
                self.std_err()
            
            if self.dist == 'normal':
                param = np.array([self.omega, self.alpha, self.beta, self.gamma])
                row_titles = ['omega', 'alpha', 'beta', 'gamma']
            if self.dist == 'student_t':
                param = np.array([self.omega, self.alpha, self.beta, self.gamma, self.nu])
                row_titles = ['omega', 'alpha', 'beta', 'gamma', 'nu']
            
            
            t_val = param/self.se
            p_val = 2*(1-norm.cdf(np.abs(t_val)))
            
            col_titles = ['', 'Estimate', 'Std. Error', 't-value', 'Pr(>|t|)', '95% CI']
            
            aux_list = []
            for i in range(len(row_titles)):
                lower = param[i]-norm.ppf(0.975)*self.se[i]/np.sqrt(self.T)
                upper = param[i]+norm.ppf(0.975)*self.se[i]/np.sqrt(self.T)
                aux_list.append([row_titles[i], round(param[i],4), round(self.se[i],4), round(t_val[i],4), p_val[i], '[%.4f, %.4f]' % (lower, upper)])
                
            print(tabulate(aux_list, headers=col_titles))
          
            if self.dist == 'normal':
                print('Loglikelihood %.2f' % self.filter(self.omega, self.alpha, self.beta, self.gamma)[2])
            if self.dist == 'student_t':
                print('Loglikelihood %.2f' % self.filter(self.omega, self.alpha, self.beta, self.gamma, self.nu)[2])
            
        
        def fit(self):
            
            # check if parameters already optimized
            if self.omega == None:
                self.mle()
            
            # run garch filter with optimized parameters
            if self.dist == 'normal':
                return self.filter(self.omega, self.alpha, self.beta, self.gamma)
            if self.dist == 'student_t':
                return self.filter(self.omega, self.alpha, self.beta, self.gamma, self.nu)
            
        def plot(self):
            
            # call fit
            fit = self.fit()
            
            # initiate figure
            fig, ax = plt.subplots(3, 1, figsize=(20,12))
            axs = ax.ravel()
            
            # plot 1-step ahead conditional variance
            ax[0].plot(self.Y.index, fit[0])
            
            # plot ret and 95% CI
            ax[1].scatter(self.Y.index, self.Y)
            if self.dist == 'normal':
                ax[1].plot(self.Y.index, norm.ppf(0.975)*np.sqrt(fit[0]), c='red')
                ax[1].plot(self.Y.index, -norm.ppf(0.975)*np.sqrt(fit[0]), c='red')
                
                # qqplot
                sm.qqplot(fit[1], line='45', ax=ax[2])
                
            if self.dist == 'student_t':
                ax[1].plot(self.Y.index, t.ppf(0.975, self.nu)*np.sqrt(fit[0]), c='red')
                ax[1].plot(self.Y.index, -t.ppf(0.975, self.nu)*np.sqrt(fit[0]), c='red')
                
                # qqplot
                sm.qqplot(fit[1], t, distargs=(self.nu,), line='45', ax=ax[2])
            
            return None
        
    class egarch:
        
        def __init__(self, Y, dist='normal'):
            self.Y = Y
            self.T = len(Y)
            self.omega = None
            self.alpha = None
            self.beta = None
            self.gamma = None
            self.nu = None
            self.hess = False
            self.se = None
            self.dist = dist
            
        def filter(self, omega, alpha, beta, gamma, nu=None):    
            log_sig2 = np.zeros(self.T)
            
            Y_abs = np.abs(self.Y)
            # initiate
            log_sig2[0] = np.log(self.T/(self.T-1)*np.var(self.Y))
            
            # define m 
            # m is equal to the expecation of the absolute of the innovations (m=sqrt(2/pi) if innovations assumed to be normal)
            m = np.sqrt(2/np.pi)
            
            # compute variance
            for t in range(1, self.T):
                vol = np.sqrt(np.exp(log_sig2[t-1]))
                log_sig2[t] = omega + alpha*(Y_abs[t-1]/vol - m) + gamma*self.Y[t-1]/vol + beta*log_sig2[t-1]
            
            # take exponential of log variance series
            sig2 = np.exp(log_sig2)
            
            # calculate standardized errors
            z = self.Y/np.sqrt(sig2)
        
            # calculate LL
            if self.dist == 'normal':
                loglik = (- 0.5*self.T*np.log(2*np.pi) - 0.5*sum(np.log(sig2)) - 0.5 * sum(self.Y**2/sig2)) 
            
            elif self.dist == 'student_t':
                term1 = self.T*np.log(G((nu+1)/2)/(np.sqrt(np.pi*(nu-2))*G(nu/2)))
                term2 = 0.5*sum(np.log(sig2))
                term3 = (nu+1)/2*sum(np.log(1+(np.sqrt(sig2)*z)**2/(sig2*(nu-2))))
                loglik = term1 - term2 - term3
                
            res = [sig2, z, loglik]
            
            return res
        
        def obj_func(self, param):
            
            # no parameter constraints (besides beta in [0,1], but that is enforced in optimization step) due to log spec.
            
            if self.dist == 'normal':
                omega, alpha, beta, gamma = param
                neg_loglik = -1*self.filter(omega, alpha, beta, gamma)[2]
                
            elif self.dist == 'student_t':
                omega, alpha, beta, gamma, nu = param
                neg_loglik = -1*self.filter(omega, alpha, beta, gamma, nu)[2]
        
            return neg_loglik
        
        def mle(self):
            
            if self.dist == 'normal':
                if self.omega != None:
                    print('Parameters already optimized: \n')
                    print('omega = %.4f, alpha = %.4f, beta = %.4f, gamma = %.4f' % (self.omega, self.alpha, self.beta, self.gamma))
                    return None
                
                res = minimize(self.obj_func, (0.1, 0.05, 0.9, 0.05),
                            bounds=[(None,None), (None,None), (1e-5,1), (None,None)], method='L-BFGS-B')
                
                opt_param = res.x
                self.omega, self.alpha, self.beta, self.gamma = opt_param
                
            if self.dist == 'student_t':
                if self.omega != None:
                    print('Parameters already optimized: \n')
                    print('omega = %.4f, alpha = %.4f, beta = %.4f, gamma = %.4f, nu = %.4f' % (self.omega, self.alpha, self.beta, self.gamma, self.nu))
                    return None

                res = minimize(self.obj_func, (0.1, 0.05, 0.9, 0.05, 8),
                            bounds=[(None,None), (None,None), (1e-5,1), (None,None), (2.0001, None)], method='L-BFGS-B')
                
                opt_param = res.x
                self.omega, self.alpha, self.beta, self.gamma, self.nu = opt_param
            
            
            return res
        
        
        
        def std_err(self, h = 1e-4):
            '''
            Function that compute the standard errors using the asymptotic variance covariance matrix of the maximum
            likelihood estimator based on the Hessian og the log-likelihood, where the Hessian is evaluated at $\hat{\theta}$.
            The Hessian is approximated numerically using finite differences. Let $\varepsilon$ be defined as $\hat{\theta}$, where 
            $\hat{\theta}$ is the maximum likelihood estimator and $h$ is the numerical step size. Let $D$ be a $3 \times 3$ diagonal
            matrix, whose diagonal is equal to $\varepsilon$. Let $\varepsilon_{i}$ denote a $3$ dimensional vector with $1$ in 
            position $i$ and zero otherwise. Then, the $(i,j)$ entry of $H_{t}(\hat{\theta})$ is approximated as
            $$
            .....
            $$
            
            '''
            if self.hess != False:
                print('Standard errors already computed: \n')
                print('omega_se = %.6f, alpha_se = %.6f, beta_se = %.6f, gamma_se = %.6f' % (self.se[0], self.se[1], self.se[2], self.se[3]))
                return None
            
                if self.dist == 'normal':
                    print('Standard errors already computed: \n')
                    print('omega_se = %.6f, alpha_se = %.6f, beta_se = %.6f, gamma_se = %.6f' % (self.se[0], self.se[1], self.se[2], self.se[3]))
                elif self.dist == 'student_t':
                    print('Standard errors already computed: \n')
                    print('omega_se = %.6f, alpha_se = %.6f, beta_se = %.6f, gamma_se = %.6f, nu_se = %.6f'
                          % (self.se[0], self.se[1], self.se[2], self.se[3], self.se[4]))
                return None
            
            if self.dist == 'normal':
                param = np.array([self.omega, self.alpha, self.beta, self.gamma])
            elif self.dist == 'student_t':
                param = np.array([self.omega, self.alpha, self.beta, self.gamma, self.nu])
                
            eps = param * h
            eps_mat = np.diag(eps)
            i_mat = np.diag(np.repeat(1,len(param)))
            hess = np.zeros([len(param),len(param)])
            
            for i in range(len(param)):
                for j in range(len(param)):
                    
                    add_term_1 = np.matmul(eps_mat, i_mat[i])
                    add_term_2 = np.matmul(eps_mat, i_mat[j])
                                            
                    new_param = param + add_term_1 + add_term_2
                    l1 = -1*self.obj_func(new_param)
                    new_param = param+ add_term_1 - add_term_2
                    l2 = -1*self.obj_func(new_param)
                    new_param = param - add_term_1 + add_term_2
                    l3 = -1*self.obj_func(new_param)
                    new_param = param - add_term_1 - add_term_2
                    l4 = -1*self.obj_func(new_param)
                    
                    hess[i][j] = (l1-l2-l3+l4) / (4*eps[i]*eps[j])
            
            hat_vcv = np.linalg.inv(-1*hess)
            se =  np.sqrt(np.diag(hat_vcv))
            self.hess = True
            self.se = se
            
            return se
        
        
        def coef(self):
            if self.omega == None:
                self.mle()
            
            if self.hess == False:
                self.std_err()
            
            if self.dist == 'normal':
                param = np.array([self.omega, self.alpha, self.beta, self.gamma])
                row_titles = ['omega', 'alpha', 'beta', 'gamma']
            if self.dist == 'student_t':
                param = np.array([self.omega, self.alpha, self.beta, self.gamma, self.nu])
                row_titles = ['omega', 'alpha', 'beta', 'gamma', 'nu']
            
            
            t_val = param/self.se
            p_val = 2*(1-norm.cdf(np.abs(t_val)))
            
            col_titles = ['', 'Estimate', 'Std. Error', 't-value', 'Pr(>|t|)', '95% CI']
            
            aux_list = []
            for i in range(len(row_titles)):
                lower = param[i]-norm.ppf(0.975)*self.se[i]/np.sqrt(self.T)
                upper = param[i]+norm.ppf(0.975)*self.se[i]/np.sqrt(self.T)
                aux_list.append([row_titles[i], round(param[i],4), round(self.se[i],4), round(t_val[i],4), p_val[i], '[%.4f, %.4f]' % (lower, upper)])
                
            print(tabulate(aux_list, headers=col_titles))
          
            if self.dist == 'normal':
                print('Loglikelihood %.2f' % self.filter(self.omega, self.alpha, self.beta, self.gamma)[2])
            if self.dist == 'student_t':
                print('Loglikelihood %.2f' % self.filter(self.omega, self.alpha, self.beta, self.gamma, self.nu)[2])
            
        
        def fit(self):
            
            # check if parameters already optimized
            if self.omega == None:
                self.mle()
            
            # run garch filter with optimized parameters
            if self.dist == 'normal':
                return self.filter(self.omega, self.alpha, self.beta, self.gamma)
            if self.dist == 'student_t':
                return self.filter(self.omega, self.alpha, self.beta, self.gamma, self.nu)
            
        def plot(self):
            
           # call fit
            fit = self.fit()
            
            # initiate figure
            fig, ax = plt.subplots(3, 1, figsize=(20,12))
            axs = ax.ravel()
            
            # plot 1-step ahead conditional variance
            ax[0].plot(self.Y.index, fit[0])
            
            # plot ret and 95% CI
            ax[1].scatter(self.Y.index, self.Y)
            if self.dist == 'normal':
                ax[1].plot(self.Y.index, norm.ppf(0.975)*np.sqrt(fit[0]), c='red')
                ax[1].plot(self.Y.index, -norm.ppf(0.975)*np.sqrt(fit[0]), c='red')
                
                # qqplot
                sm.qqplot(fit[1], line='45', ax=ax[2])
                
            if self.dist == 'student_t':
                ax[1].plot(self.Y.index, t.ppf(0.975, self.nu)*np.sqrt(fit[0]), c='red')
                ax[1].plot(self.Y.index, -t.ppf(0.975, self.nu)*np.sqrt(fit[0]), c='red')
                
                # qqplot
                sm.qqplot(fit[1], t, distargs=(self.nu,), line='45', ax=ax[2])
            
            return None
