import logging

import numpy as np
from scipy.stats import chi2
from scipy.stats import norm
from scipy.stats import t

LOG = logging.getLogger('stat_utilities')
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s - %(name)s - %(message)s')


def __get_alpha(alpha):
    if alpha >= 1.0 or alpha <= 0:
        LOG.warning("alpha value outside of (0, 1) has been truncated to default value of 0.05")
        return 0.05
    return alpha


def ci_small_n(vals, alpha=0.05):
    """
    A function to calculate a Confidence Interval when the population is assumed
    to be Normal, Sigma is unknown, and the sample size is small

    Args:
        vals(iterable): An iterable of float values
        alpha(float): The significance level in (0, 1), default 0.05

    Returns:
        The Confidence Interval at significance level alpha for the population mean
    """
    alpha = __get_alpha(alpha)
    x_bar = np.mean(vals)
    s = np.std(vals)
    n = len(vals)
    if n > 25:
        LOG.warning("Consider using ci_large_n with n > 25")
    width = t.ppf(1 - (alpha/2.0), df=n-1) * (s/np.sqrt(n))
    return x_bar - width, x_bar + width


# Use when given sample mean, sample standard deviation, and Normal distribution with large n
def ci_large_n(vals, alpha=0.05):
    """
    A function to calculate a Confidence Interval when the population is assumed
    to be Normal, Sigma is unknown, and the sample size is large

    Args:
        vals(iterable): An iterable of float values
        alpha(float): The significance level in (0, 1), default 0.05

    Returns:
        The Confidence Interval at significance level alpha for the population mean
    """
    alpha = __get_alpha(alpha)
    x_bar = np.mean(vals)
    s = np.std(vals)
    n = len(vals)
    if n <= 25:
        LOG.warning("ci_small_n should be used with n <= 25")
    width = norm.ppf(1 - (alpha/2.0)) * (s/np.sqrt(n))
    return x_bar - width, x_bar + width


def ci_sigma(vals, sigma, alpha=0.05):
    """
    Calculates a Confidence Interval when the population is assumed
    to be Normal, Sigma is known, and the sample size is large

    Args:
        vals(iterable): An iterable of float values
        alpha(float): The significance level in (0, 1), default 0.05
        sigma(float): The population standard deviation

    Returns:
        The Confidence Interval at significance level alpha for the population mean
    """
    alpha = __get_alpha(alpha)
    x_bar = np.mean(vals)
    n = len(vals)
    width = norm.ppf(1 - (alpha/2.0)) * (sigma/np.sqrt(n))
    return x_bar - width, x_bar + width


# Use when given a proportion with a sample mean
def proportion_ci(p_hat, n, alpha=0.05):
    """
    Calculates a proportion Confidence Interval

    Args:
        p_hat(float): The sample proportion
        alpha(float): The significance level in (0, 1), default 0.05
        n(int): The sample size

    Returns:
        The Confidence Interval at significance level alpha for the proportion
    """
    alpha = __get_alpha(alpha)
    width = norm.ppf(1 - (alpha/2.0)) * (np.sqrt((p_hat * (1.0 - p_hat))/n))
    return p_hat - width, p_hat + width


# Use when calculating confidence interval for variance
def variance_ci(vals, alpha=0.05):
    """
    Calculates a Confidence Interval for the population variance for the
    given values at significance level alpha

    Args:
        vals(iterable): An iterable of float values
        alpha(float): The significance level in (0, 1), default 0.05

    Return:
        The Confidence Interval at significance level alpha for Sigma
    """
    alpha = __get_alpha(alpha)
    n = len(vals)
    x_bar = np.mean(vals)
    s_squared = np.sum(np.power(vals - x_bar, 2))/(n - 1)
    denom_1 = chi2.ppf(1 - alpha/2.0, df=n-1)
    denom_2 = chi2.ppf((alpha/2.0), df=n-1)
    num = (n - 1) * s_squared
    return num/denom_1, num/denom_2


# Use when comparing difference of means for normal populations with sigma known
def mean_diff_ci_sigma(sample_1, sample_2, sigma_1, sigma_2, alpha=0.05):
    """
    Calculates a Confidence Interval at significance level alpha for the
    difference of means of two samples drawn from Normal distributions
    when Sigma is known for both samples
    Args:
        sample_1(iterable): An iterable of float values
        sample_2(iterable): An iterable of float values
        sigma_1(float): The population variance for sample_1
        sigma_2(float): The population variance for sample_2
        alpha(float): The significance level in (0, 1), default 0.05

    Returns:
        The Confidence Interval at significance level alpha for the difference
        in means
    """
    x_bar1 = np.mean(sample_1)
    x_bar2 = np.mean(sample_2)
    n_1 = len(sample_1)
    n_2 = len(sample_2)
    alpha = __get_alpha(alpha)
    mean_diff = x_bar1 - x_bar2
    width = norm.ppf(1 - (alpha/2.0)) * np.sqrt((np.power(sigma_1, 2)/n_1) + (np.power(sigma_2, 2)/n_2))
    return mean_diff - width, mean_diff + width


# Use when comparing difference of means for non-normal populations, population sigma unknown, large n
def mean_diff_ci_large_n(sample_1, sample_2, alpha=0.05):
    """
    Calculates a Confidence Interval at significance level alpha for the
    difference of means of two samples drawn from non-Normal distributions
    when Sigma is unknown for both samples and n is large (i.e., > 25)
    Args:
        sample_1(iterable): An iterable of float values
        sample_2(iterable): An iterable of float values
        alpha(float): The significance level in (0, 1), default 0.05

    Returns:
        The Confidence Interval at significance level alpha for the difference
        in means
    """
    alpha = __get_alpha(alpha)
    x_bar1 = np.mean(sample_1)
    x_bar2 = np.mean(sample_2)
    n_1 = len(sample_1)
    n_2 = len(sample_2)
    std_1 = np.std(sample_1)
    std_2 = np.std(sample_2)
    mean_diff = x_bar1 - x_bar2
    width = norm.ppf(1 - (alpha/2.0)) * np.sqrt((np.power(std_1, 2)/n_1) + (np.power(std_2, 2)/n_2))
    return mean_diff - width, mean_diff + width


# Use when comparing difference of means for non-normal populations, population sigma unknown, small n
def mean_diff_ci_small_n(sample_1, sample_2, alpha=0.05):
    """
    Calculates a Confidence Interval at significance level alpha for the
    difference of means of two samples drawn from non-Normal distributions
    when Sigma is unknown for both samples and n is small (i.e., <= 25)
    Args:
        sample_1(iterable): An iterable of float values
        sample_2(iterable): An iterable of float values
        alpha(float): The significance level in (0, 1), default 0.05

    Returns:
        The Confidence Interval at significance level alpha for the difference
        in means
    """
    alpha = __get_alpha(alpha)
    x_bar1 = np.mean(sample_1)
    x_bar2 = np.mean(sample_2)
    n_1 = len(sample_1)
    n_2 = len(sample_2)
    std_1 = np.std(sample_1)
    std_2 = np.std(sample_2)
    mean_diff = x_bar1 - x_bar2
    se_1 = std_1/np.sqrt(n_1)
    se_2 = std_2/np.sqrt(n_2)
    v = (np.power((np.power(se_1, 2) + np.power(se_2, 2)), 2) / 
        ((np.power(se_1, 4)/(n_1 - 1)) + (np.power(se_2, 4)/(n_2 - 1))))
    width = t.ppf(1 - (alpha/2.0), df=np.floor(v)) * (np.sqrt((np.power(std_1, 2)/n_1) + (np.power(std_2, 2)/n_2)))
    return mean_diff - width, mean_diff + width
