from typing import Any, Dict

import numpy as np
import pandas as pd


def constant_corr_shrinkage(
    returns: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Calculate the shrinkage target, the optimal shrinkage intensity and the shrunk covariance
    matrix for the constant correlation shrinkage. No check is done on the input data.
    Implementation follows the Ledoit-Wolf (2003) paper "Honey, I Shrunk the Sample Covariance
    Matrix".

    Parameters:
        returns (pd.DataFrame): The returns matrix (T x N) where T is time periods and N is assets

    Returns:
        Dict[str, Any]: A dictionary containing the shrinkage target matrix ("target"), the optimal
            shrinkage intensity ("intensity"), the sample covariance matrix ("sample_cov") and the
            shrunk covariance matrix ("shrunk_cov").
    """

    cov_matrix = returns.cov()

    T, N = returns.shape  # T = time periods, N = number of assets

    # Convert to numpy arrays for efficiency
    returns_np = returns.values
    S = cov_matrix.values  # Sample covariance matrix

    # Extract standard deviations
    variances = np.diag(S)
    std_devs = np.sqrt(variances)  # fixed it

    # Compute correlation matrix from covariance matrix
    std_outer = np.outer(std_devs, std_devs)
    corr_matrix = S / std_outer

    # Calculate average correlation
    avg_corr = (
        2 / (N * (N - 1)) * np.triu(corr_matrix, k=1).sum()
    )  # !!! COMPLETE AS APPROPRIATE !!!

    ## Target
    # Calculate target matrix (constant correlation)
    constant_corr_cov = avg_corr * np.outer(
        std_devs, std_devs
    )  # !!! COMPLETE AS APPROPRIATE !!!

    # Set diagonal elements to original variances
    np.fill_diagonal(constant_corr_cov, variances)

    target = pd.DataFrame(
        constant_corr_cov, index=cov_matrix.index, columns=cov_matrix.columns
    )

    ## Intensity
    sample_means = np.mean(returns_np, axis=0)

    # Center the returns
    centered_returns = returns_np - sample_means

    # Calculate pi-hat and rho-hat using vectorized operations
    # Pre-compute matrices for efficiency
    sqrt_ratio_matrix = np.outer(std_devs, 1 / std_devs)  # sqrt(S[i,i]/S[j,j])

    pi_hat = 0.0
    rho_hat = 0.0

    # Vectorized for loop calculations
    centered = returns_np - sample_means  # (T x N) centered returns
    sum_outer_sq = (centered**2).T @ (centered**2)  # sum_t y_ti^2 * y_tj^2
    sum_outer = (T - 1) * S  # sum_t y_ti * y_tj (since cov uses T-1)
    pi_hat = (
        np.sum(sum_outer_sq) - 2 * np.sum(S * sum_outer) + T * np.sum(S**2)
    )  # we divide by T after

    # rho_hat diagonal
    sum_y4 = np.diag(sum_outer_sq)
    rho_diag = (
        np.sum(sum_y4)
        - 2 * np.sum(variances * np.diag(sum_outer))
        + T * np.sum(variances**2)
    )  # we divide by T after

    # rho_hat off-diagonal
    c3 = centered**3
    M3 = c3.T @ centered  # M3[i,j]  = sum_t y_ti^3 * y_tj
    M3T = centered.T @ c3  # M3T[i,j] = sum_t y_ti * y_tj^3

    var_row = variances[:, np.newaxis]
    var_col = variances[np.newaxis, :]

    theta_ii_ij = M3 - (T - 2) * var_row * S  # we divide by T after
    theta_jj_ij = M3T - (T - 2) * var_col * S

    off_diag = (avg_corr / 2) * (
        sqrt_ratio_matrix.T * theta_ii_ij + sqrt_ratio_matrix * theta_jj_ij
    )
    np.fill_diagonal(off_diag, 0)

    rho_hat = rho_diag + np.sum(off_diag)

    pi_hat /= T  # ERROR: took out from the for loop
    rho_hat /= T  # ERROR: took out from the for loop

    # Calculate gamma-hat: ||F - S||^2 where F is the target matrix
    gamma_hat = np.sum((target.values - S) ** 2)

    # Calculate optimal shrinkage intensity
    # k = (pi - rho) / gamma, but we need to be careful about numerical stability
    numerator = pi_hat - rho_hat
    denominator = gamma_hat

    if denominator == 0:
        intensity = 0.0
    else:
        # Ensure shrinkage intensity is between 0 and 1
        intensity = max(
            0.0, min(1.0, numerator / (T * denominator))
        )  # ERROR: added T in the denominator

    return {
        "target": target,
        "intensity": intensity,
        "sample_cov": cov_matrix,
        "shrunk_cov": intensity * target
        + (1 - intensity)
        * cov_matrix,  # ERROR: corrected the formula for shrunk covariance
    }


def market_factor_shrinkage(
    returns: pd.DataFrame, market_returns: pd.Series
) -> Dict[str, Any]:
    """
    Calculate the shrinkage target, the optimal shrinkage intensity and the shrunk covariance
    matrix for the market factor shrinkage. No check is done on the input data.
    Implementation follows the Ledoit-Wolf (2002) paper "Improved estimation of the covariance
    matrix of stock returns with an application to portfolio selection".

    Parameters:
        returns (pd.DataFrame): The returns matrix (T x N) where T is time periods and N is assets
        market_returns (pd.Series): The market returns (T x 1)

    Returns:
        Dict[str, Any]: A dictionary containing the shrinkage target matrix ("target"), the optimal
            shrinkage intensity ("intensity"), the sample covariance matrix ("sample_cov") and the
            shrunk covariance matrix ("shrunk_cov").
    """

    # Align indices to ensure proper calculation
    aligned_data = pd.concat([returns, market_returns], axis=1, join="inner")
    returns_aligned = aligned_data.iloc[:, :-1]
    market_aligned = aligned_data.iloc[:, -1]

    T = returns_aligned.shape[0]

    ## Target
    # Calculate market variance
    market_variance = market_aligned.var()  # ERROR: changed from std to var

    # Calculate betas for all assets (vectorized)
    returns_np = returns_aligned.values
    market_np = market_aligned.values

    # Compute covariances between all assets and market at once
    # Stack returns and market, compute covariance matrix, extract asset-market covariances
    combined = np.column_stack([returns_np, market_np])
    cov_matrix_full = np.cov(combined.T)
    cov_with_market = cov_matrix_full[:-1, -1]  # Covariances of each asset with market
    betas = np.divide(cov_with_market, market_variance)

    # Calculate residual variances: Var(asset) - Beta² * Var(market) (vectorized)
    asset_variances = np.diag(
        cov_matrix_full[:-1, :-1]
    )  # remove last row and column because they correspond to the market itself
    residual_variances = asset_variances - betas**2 * market_variance

    # Ensure residual variances are positive (vectorized)
    residual_variances = np.maximum(residual_variances, 1e-8)

    # Construct target matrix
    betas_outer = np.outer(betas, betas)
    residual_matrix = np.diag(residual_variances)

    # Final target matrix
    target = pd.DataFrame(
        market_variance * np.outer(betas, betas) + residual_matrix,
        index=returns.columns,
        columns=returns.columns,
    )

    ## Intensity
    cov_matrix = returns_aligned.cov()
    S = cov_matrix.values

    # Center the returns
    sample_means = np.mean(returns_np, axis=0)
    centered_returns = returns_np - sample_means
    market_centered_returns = market_np - market_np.mean()

    # Calculate pi-hat and rho-hat
    # Pre-compute matrices for efficiency
    variances = np.diag(S)
    target_np = target.values

    pi_hat = 0.0
    rho_hat = 0.0

    for t in range(T):
        y_t = centered_returns[t, :]
        outer_prod = np.outer(y_t, y_t)
        diff = outer_prod - S  # ERROR: changed sign as before

        # Pi-hat calculation
        pi_hat += np.sum(diff**2)  # ERROR: added + to sum at each iteration

        # Rho-hat calculation
        m_t = market_centered_returns[t]

        # Diagonal terms: (y_t[i]**2 - S[i,i])**2
        diag_terms = y_t**2 - variances
        rho_hat += np.sum(diag_terms**2)

        # Off-diagonal terms
        # Create matrices: betas[j] * y_t[i] for all i,j
        betas_y_matrix = np.outer(y_t, betas)  # y_t[i] * betas[j]
        betas_y_matrix_T = betas_y_matrix.T  # y_t[j] * betas[i]

        # Calculate the rho-hat contribution for off-diagonal elements
        off_diag_term = (
            betas_y_matrix_T + betas_y_matrix - betas_outer * m_t
        ) * m_t * outer_prod - target_np * S

        # Sum only off-diagonal elements
        np.fill_diagonal(off_diag_term, 0)
        rho_hat += np.sum(off_diag_term)

    rho_hat /= T
    pi_hat /= T  # ERROR: added this line

    # Calculate gamma-hat: ||F - S||^2 where F is the target matrix
    gamma_hat = np.sum((target_np - S) ** 2)

    # Calculate optimal shrinkage intensity
    # k = (pi - rho) / gamma, but we need to be careful about numerical stability
    numerator = pi_hat - rho_hat
    denominator = gamma_hat

    if denominator == 0:
        intensity = 0.0
    else:
        intensity = max(0.0, min(1.0, numerator / (T * denominator)))

    return {
        "target": target,
        "intensity": intensity,
        "sample_cov": cov_matrix,
        "shrunk_cov": intensity * target
        + (1 - intensity)
        * cov_matrix,  # ERROR: corrected the formula for shrunk covariance
    }
