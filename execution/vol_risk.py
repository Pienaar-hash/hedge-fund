"""
v7.5_A1 — Parametric VaR + Position CVaR Risk Module

Provides institutional risk metrics:
- Portfolio VaR using EWMA covariance matrix (parametric approach)
- Position CVaR (Expected Shortfall at configurable confidence level)

These integrate with the risk engine for veto authority.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger("vol_risk")

# Default paths
DEFAULT_STATE_DIR = Path(os.getenv("HEDGE_STATE_DIR") or "logs/state")
DEFAULT_OHLCV_DIR = Path("data/ohlcv")

# Z-score lookup table for common confidence levels (avoids scipy dependency)
# These are norm.ppf(confidence) values
_Z_SCORE_TABLE = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.975: 1.9600,
    0.99: 2.3263,
    0.995: 2.5758,
    0.999: 3.0902,
}


def _norm_ppf(confidence: float) -> float:
    """
    Return the z-score (standard normal quantile) for a given confidence level.
    Uses lookup table for common values, falls back to approximation for others.
    """
    # Check lookup table first
    if confidence in _Z_SCORE_TABLE:
        return _Z_SCORE_TABLE[confidence]
    
    # For other values, use a rational approximation (Abramowitz & Stegun 26.2.23)
    # This is accurate to ~4.5e-4 for the range 0.5 < p < 1
    if confidence <= 0.5:
        return -_norm_ppf(1.0 - confidence)
    
    p = confidence
    # Intermediate value
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    
    # Coefficients for rational approximation
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    
    z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    return z


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class VaRConfig:
    """Configuration for Portfolio VaR calculation."""
    enabled: bool = True
    confidence: float = 0.99
    lookback_bars: int = 500
    halflife_bars: int = 100
    max_portfolio_var_nav_pct: float = 0.12


@dataclass
class CVaRConfig:
    """Configuration for Position CVaR calculation."""
    enabled: bool = True
    confidence: float = 0.95
    lookback_bars: int = 400
    max_position_cvar_nav_pct: float = 0.04


@dataclass
class VaRResult:
    """Result of Portfolio VaR calculation."""
    var_usd: float
    var_nav_pct: float
    portfolio_volatility: float  # Annualized portfolio volatility
    weights: Dict[str, float] = field(default_factory=dict)
    n_assets: int = 0
    lookback_used: int = 0
    confidence: float = 0.99
    within_limit: bool = True
    limit_nav_pct: float = 0.12


@dataclass
class CVaRResult:
    """Result of Position CVaR calculation."""
    symbol: str
    cvar_usd: float
    cvar_nav_pct: float
    var_usd: float  # Also compute VaR for reference
    var_nav_pct: float
    position_notional_usd: float
    confidence: float = 0.95
    lookback_used: int = 0
    within_limit: bool = True
    limit_nav_pct: float = 0.04


@dataclass
class RiskAdvancedSnapshot:
    """Combined risk snapshot for state publishing."""
    var: Optional[VaRResult] = None
    cvar_by_symbol: Dict[str, CVaRResult] = field(default_factory=dict)
    updated_ts: float = 0.0


# ---------------------------------------------------------------------------
# EWMA Covariance Matrix
# ---------------------------------------------------------------------------

def compute_ewma_weights(n: int, halflife: int) -> np.ndarray:
    """
    Compute exponentially weighted moving average weights.
    
    Args:
        n: Number of observations
        halflife: Halflife in number of observations
        
    Returns:
        Array of weights (most recent observation has highest weight)
    """
    if halflife <= 0:
        # Equal weights fallback
        return np.ones(n) / n
    
    # Lambda decay factor: lambda = 0.5^(1/halflife)
    decay = 0.5 ** (1.0 / halflife)
    
    # Weights: w_i = (1-lambda) * lambda^i for i = 0, 1, ..., n-1
    # Newest observation is at index n-1, so we reverse
    weights = np.array([(1 - decay) * (decay ** i) for i in range(n)])
    weights = weights[::-1]  # Reverse so most recent has highest weight
    
    # Normalize to sum to 1
    weights = weights / weights.sum()
    
    return weights


def compute_ewma_covariance(
    returns_matrix: np.ndarray,
    halflife_bars: int = 100,
) -> np.ndarray:
    """
    Compute EWMA covariance matrix from returns.
    
    Args:
        returns_matrix: Shape (N_assets, T) array of returns
        halflife_bars: Halflife for EWMA weighting
        
    Returns:
        Covariance matrix of shape (N_assets, N_assets)
    """
    n_assets, n_obs = returns_matrix.shape
    
    if n_obs < 2:
        LOG.warning("[vol_risk] insufficient observations for covariance: %d", n_obs)
        return np.eye(n_assets) * 0.01  # Default 1% daily vol
    
    # Compute EWMA weights
    weights = compute_ewma_weights(n_obs, halflife_bars)
    
    # Demean returns using weighted mean
    weighted_means = np.zeros(n_assets)
    for i in range(n_assets):
        weighted_means[i] = np.sum(weights * returns_matrix[i, :])
    
    demeaned = returns_matrix - weighted_means.reshape(-1, 1)
    
    # Compute weighted covariance
    # Cov[i,j] = sum(w_t * r_i,t * r_j,t)
    cov_matrix = np.zeros((n_assets, n_assets))
    for t in range(n_obs):
        outer = np.outer(demeaned[:, t], demeaned[:, t])
        cov_matrix += weights[t] * outer
    
    return cov_matrix


# ---------------------------------------------------------------------------
# Portfolio VaR (Parametric)
# ---------------------------------------------------------------------------

def compute_portfolio_var(
    returns_matrix: np.ndarray,
    weights: np.ndarray,
    nav_usd: float,
    confidence: float = 0.99,
    halflife_bars: int = 100,
    holding_period_days: float = 1.0,
) -> VaRResult:
    """
    Compute parametric Portfolio VaR using EWMA covariance.
    
    Args:
        returns_matrix: Shape (N_assets, T) array of daily returns
        weights: Shape (N_assets,) portfolio weights (fraction of NAV)
        nav_usd: Total portfolio NAV in USD
        confidence: VaR confidence level (e.g., 0.99 for 99%)
        halflife_bars: EWMA halflife for covariance estimation
        holding_period_days: VaR horizon in days (default 1-day VaR)
        
    Returns:
        VaRResult with computed VaR metrics
    """
    n_assets = len(weights)
    
    if returns_matrix.shape[0] != n_assets:
        LOG.error(
            "[vol_risk] returns shape mismatch: expected %d assets, got %d",
            n_assets,
            returns_matrix.shape[0],
        )
        return VaRResult(
            var_usd=0.0,
            var_nav_pct=0.0,
            portfolio_volatility=0.0,
            n_assets=n_assets,
            lookback_used=0,
            confidence=confidence,
        )
    
    n_obs = returns_matrix.shape[1]
    
    # Compute EWMA covariance matrix
    cov_matrix = compute_ewma_covariance(returns_matrix, halflife_bars)
    
    # Portfolio variance: w^T * Sigma * w
    portfolio_variance = float(weights @ cov_matrix @ weights)
    
    # Portfolio standard deviation (daily)
    portfolio_std = math.sqrt(max(0.0, portfolio_variance))
    
    # Scale for holding period (assuming i.i.d.)
    scaled_std = portfolio_std * math.sqrt(holding_period_days)
    
    # Z-score for confidence level
    # For 99% confidence, z ≈ 2.326
    # For 95% confidence, z ≈ 1.645
    z_score = _norm_ppf(confidence)
    
    # VaR in USD
    var_usd = z_score * scaled_std * nav_usd
    
    # VaR as percentage of NAV
    var_nav_pct = z_score * scaled_std if nav_usd > 0 else 0.0
    
    # Annualized portfolio volatility (for reporting)
    annualized_vol = portfolio_std * math.sqrt(252)
    
    return VaRResult(
        var_usd=var_usd,
        var_nav_pct=var_nav_pct,
        portfolio_volatility=annualized_vol,
        n_assets=n_assets,
        lookback_used=n_obs,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Position CVaR (Expected Shortfall)
# ---------------------------------------------------------------------------

def compute_position_cvar(
    returns: np.ndarray,
    nav_usd: float,
    position_notional_usd: float,
    confidence: float = 0.95,
) -> CVaRResult:
    """
    Compute Expected Shortfall (CVaR) for a single position.
    
    CVaR is the expected loss given that loss exceeds VaR.
    It's more conservative than VaR as it accounts for tail risk.
    
    Args:
        returns: 1D array of historical returns for the position
        nav_usd: Total portfolio NAV in USD
        position_notional_usd: Position size in USD
        confidence: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        CVaRResult with computed metrics
    """
    if len(returns) < 10:
        LOG.warning("[vol_risk] insufficient returns for CVaR: %d", len(returns))
        return CVaRResult(
            symbol="",
            cvar_usd=0.0,
            cvar_nav_pct=0.0,
            var_usd=0.0,
            var_nav_pct=0.0,
            position_notional_usd=position_notional_usd,
            confidence=confidence,
            lookback_used=len(returns),
        )
    
    # Compute PnL distribution = notional * returns
    pnl_distribution = position_notional_usd * returns
    
    # Sort PnL (worst losses first - most negative)
    sorted_pnl = np.sort(pnl_distribution)
    
    # VaR: (1-confidence) quantile of losses
    # For 95% confidence, we look at worst 5% of outcomes
    var_quantile = 1 - confidence
    var_index = int(len(sorted_pnl) * var_quantile)
    var_index = max(1, var_index)  # At least 1 observation
    
    # VaR is the threshold loss (negative value = loss)
    var_usd = -sorted_pnl[var_index - 1]  # Negate to get positive loss value
    
    # CVaR: Mean of losses worse than VaR (expected shortfall)
    tail_losses = sorted_pnl[:var_index]
    cvar_usd = -np.mean(tail_losses)  # Negate to get positive expected loss
    
    # Convert to NAV percentages
    var_nav_pct = var_usd / nav_usd if nav_usd > 0 else 0.0
    cvar_nav_pct = cvar_usd / nav_usd if nav_usd > 0 else 0.0
    
    return CVaRResult(
        symbol="",
        cvar_usd=max(0.0, cvar_usd),
        cvar_nav_pct=max(0.0, cvar_nav_pct),
        var_usd=max(0.0, var_usd),
        var_nav_pct=max(0.0, var_nav_pct),
        position_notional_usd=position_notional_usd,
        confidence=confidence,
        lookback_used=len(returns),
    )


# ---------------------------------------------------------------------------
# Config Loaders
# ---------------------------------------------------------------------------

def load_var_config(strategy_config: Mapping[str, Any] | None = None) -> VaRConfig:
    """Load VaR configuration from strategy config."""
    if strategy_config is None:
        try:
            with open("config/strategy_config.json") as f:
                strategy_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return VaRConfig()
    
    risk_adv = strategy_config.get("risk_advanced", {})
    var_cfg = risk_adv.get("var", {})
    
    return VaRConfig(
        enabled=bool(var_cfg.get("enabled", True)),
        confidence=float(var_cfg.get("confidence", 0.99)),
        lookback_bars=int(var_cfg.get("lookback_bars", 500)),
        halflife_bars=int(var_cfg.get("halflife_bars", 100)),
        max_portfolio_var_nav_pct=float(var_cfg.get("max_portfolio_var_nav_pct", 0.12)),
    )


def load_cvar_config(strategy_config: Mapping[str, Any] | None = None) -> CVaRConfig:
    """Load CVaR configuration from strategy config."""
    if strategy_config is None:
        try:
            with open("config/strategy_config.json") as f:
                strategy_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return CVaRConfig()
    
    risk_adv = strategy_config.get("risk_advanced", {})
    cvar_cfg = risk_adv.get("cvar", {})
    
    return CVaRConfig(
        enabled=bool(cvar_cfg.get("enabled", True)),
        confidence=float(cvar_cfg.get("confidence", 0.95)),
        lookback_bars=int(cvar_cfg.get("lookback_bars", 400)),
        max_position_cvar_nav_pct=float(cvar_cfg.get("max_position_cvar_nav_pct", 0.04)),
    )


# ---------------------------------------------------------------------------
# Returns Data Loading
# ---------------------------------------------------------------------------

def load_symbol_returns(
    symbol: str,
    lookback_bars: int = 500,
    ohlcv_dir: Path | str = DEFAULT_OHLCV_DIR,
) -> np.ndarray:
    """
    Load historical returns for a symbol from OHLCV data.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        lookback_bars: Number of bars to load
        ohlcv_dir: Directory containing OHLCV files
        
    Returns:
        1D numpy array of log returns
    """
    ohlcv_dir = Path(ohlcv_dir)
    
    # Try JSON file patterns first (legacy format)
    json_patterns = [
        f"{symbol.upper()}_1d.json",
        f"{symbol.upper()}_4h.json",
        f"{symbol.upper()}_1h.json",
        f"{symbol.lower()}_1d.json",
    ]
    
    for pattern in json_patterns:
        filepath = ohlcv_dir / pattern
        if filepath.exists():
            try:
                with open(filepath) as f:
                    data = json.load(f)
                
                # Extract close prices
                if isinstance(data, list):
                    closes = [float(bar.get("close", bar.get("c", 0))) for bar in data[-lookback_bars:]]
                elif isinstance(data, dict) and "candles" in data:
                    closes = [float(bar.get("close", bar.get("c", 0))) for bar in data["candles"][-lookback_bars:]]
                else:
                    continue
                
                if len(closes) < 10:
                    continue
                
                # Compute log returns
                closes_arr = np.array(closes)
                returns = np.diff(np.log(closes_arr))
                
                return returns
                
            except Exception as exc:
                LOG.debug("[vol_risk] failed to load JSON %s: %s", filepath, exc)
                continue
    
    # Try parquet files in subdirectory structure (e.g., data/ohlcv/BTCUSDT/4h/*.parquet)
    try:
        import pandas as pd  # type: ignore
        
        symbol_dir = ohlcv_dir / symbol.upper()
        if not symbol_dir.exists():
            LOG.debug("[vol_risk] no OHLCV directory for %s", symbol)
            return np.array([])
        
        # Prefer longer timeframes for VaR (4h or 1h)
        for tf in ["4h", "1h", "15m"]:
            tf_dir = symbol_dir / tf
            if not tf_dir.exists():
                continue
            
            parquet_files = sorted(tf_dir.glob("*.parquet"))
            if not parquet_files:
                continue
            
            # Load recent parquet files
            dfs = []
            for pf in parquet_files[-30:]:  # Last 30 days
                try:
                    df = pd.read_parquet(pf)
                    dfs.append(df)
                except Exception:
                    continue
            
            if not dfs:
                continue
            
            combined = pd.concat(dfs, ignore_index=True)
            
            # Get close prices
            close_col = None
            for col in ["close", "Close", "c"]:
                if col in combined.columns:
                    close_col = col
                    break
            
            if close_col is None:
                continue
            
            closes = combined[close_col].dropna().values[-lookback_bars:]
            
            if len(closes) < 20:
                continue
            
            # Compute log returns
            returns = np.diff(np.log(closes.astype(float)))
            LOG.debug("[vol_risk] loaded %d returns for %s from parquet", len(returns), symbol)
            return returns
            
    except ImportError:
        LOG.debug("[vol_risk] pandas not available for parquet loading")
    except Exception as exc:
        LOG.debug("[vol_risk] failed to load parquet for %s: %s", symbol, exc)
    
    LOG.warning("[vol_risk] no OHLCV data found for %s", symbol)
    return np.array([])


def load_portfolio_returns(
    symbols: Sequence[str],
    lookback_bars: int = 500,
    ohlcv_dir: Path | str = DEFAULT_OHLCV_DIR,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load returns matrix for multiple symbols.
    
    Args:
        symbols: List of trading pairs
        lookback_bars: Number of bars to load
        ohlcv_dir: Directory containing OHLCV files
        
    Returns:
        Tuple of (returns_matrix, valid_symbols)
        returns_matrix has shape (n_valid_symbols, min_common_length)
    """
    returns_list = []
    valid_symbols = []
    
    for symbol in symbols:
        returns = load_symbol_returns(symbol, lookback_bars, ohlcv_dir)
        if len(returns) >= 20:  # Minimum for meaningful covariance
            returns_list.append(returns)
            valid_symbols.append(symbol)
    
    if not returns_list:
        return np.array([]).reshape(0, 0), []
    
    # Find common length (minimum)
    min_len = min(len(r) for r in returns_list)
    
    # Truncate all to common length (most recent observations)
    returns_matrix = np.array([r[-min_len:] for r in returns_list])
    
    return returns_matrix, valid_symbols


# ---------------------------------------------------------------------------
# High-Level Risk Functions
# ---------------------------------------------------------------------------

def compute_portfolio_var_from_positions(
    positions: Sequence[Mapping[str, Any]],
    nav_usd: float,
    var_config: VaRConfig | None = None,
    ohlcv_dir: Path | str = DEFAULT_OHLCV_DIR,
) -> VaRResult:
    """
    Compute Portfolio VaR from current positions.
    
    Args:
        positions: List of position dicts with 'symbol' and 'notional' keys
        nav_usd: Total portfolio NAV
        var_config: VaR configuration
        ohlcv_dir: Directory for OHLCV data
        
    Returns:
        VaRResult
    """
    if var_config is None:
        var_config = load_var_config()
    
    if not var_config.enabled:
        return VaRResult(
            var_usd=0.0,
            var_nav_pct=0.0,
            portfolio_volatility=0.0,
        )
    
    # Extract symbols and weights
    symbols = []
    notionals = []
    
    for pos in positions:
        symbol = pos.get("symbol")
        notional = abs(float(pos.get("notional", pos.get("positionAmt", 0)) or 0))
        
        if symbol and notional > 0:
            symbols.append(symbol)
            notionals.append(notional)
    
    if not symbols or nav_usd <= 0:
        return VaRResult(
            var_usd=0.0,
            var_nav_pct=0.0,
            portfolio_volatility=0.0,
        )
    
    # Compute weights as fraction of NAV
    weights = np.array(notionals) / nav_usd
    
    # Load returns matrix
    returns_matrix, valid_symbols = load_portfolio_returns(
        symbols,
        var_config.lookback_bars,
        ohlcv_dir,
    )
    
    if returns_matrix.size == 0:
        LOG.warning("[vol_risk] no returns data available for VaR calculation")
        return VaRResult(
            var_usd=0.0,
            var_nav_pct=0.0,
            portfolio_volatility=0.0,
        )
    
    # Filter weights to match valid symbols
    valid_weights = []
    symbol_weights = {}
    for i, sym in enumerate(symbols):
        if sym in valid_symbols:
            idx = valid_symbols.index(sym)
            valid_weights.append(weights[i])
            symbol_weights[sym] = float(weights[i])
    
    if not valid_weights:
        return VaRResult(
            var_usd=0.0,
            var_nav_pct=0.0,
            portfolio_volatility=0.0,
        )
    
    weights_arr = np.array(valid_weights)
    
    # Compute VaR
    result = compute_portfolio_var(
        returns_matrix,
        weights_arr,
        nav_usd,
        var_config.confidence,
        var_config.halflife_bars,
    )
    
    result.weights = symbol_weights
    result.within_limit = result.var_nav_pct <= var_config.max_portfolio_var_nav_pct
    result.limit_nav_pct = var_config.max_portfolio_var_nav_pct
    
    return result


def compute_position_cvar_for_symbol(
    symbol: str,
    position_notional_usd: float,
    nav_usd: float,
    cvar_config: CVaRConfig | None = None,
    ohlcv_dir: Path | str = DEFAULT_OHLCV_DIR,
) -> CVaRResult:
    """
    Compute CVaR for a single position.
    
    Args:
        symbol: Trading pair
        position_notional_usd: Position size in USD
        nav_usd: Total portfolio NAV
        cvar_config: CVaR configuration
        ohlcv_dir: Directory for OHLCV data
        
    Returns:
        CVaRResult
    """
    if cvar_config is None:
        cvar_config = load_cvar_config()
    
    if not cvar_config.enabled:
        return CVaRResult(
            symbol=symbol,
            cvar_usd=0.0,
            cvar_nav_pct=0.0,
            var_usd=0.0,
            var_nav_pct=0.0,
            position_notional_usd=position_notional_usd,
        )
    
    # Load returns
    returns = load_symbol_returns(symbol, cvar_config.lookback_bars, ohlcv_dir)
    
    if len(returns) < 20:
        LOG.warning("[vol_risk] insufficient returns for %s CVaR: %d bars", symbol, len(returns))
        return CVaRResult(
            symbol=symbol,
            cvar_usd=0.0,
            cvar_nav_pct=0.0,
            var_usd=0.0,
            var_nav_pct=0.0,
            position_notional_usd=position_notional_usd,
            lookback_used=len(returns),
        )
    
    result = compute_position_cvar(
        returns,
        nav_usd,
        position_notional_usd,
        cvar_config.confidence,
    )
    
    result.symbol = symbol
    result.within_limit = result.cvar_nav_pct <= cvar_config.max_position_cvar_nav_pct
    result.limit_nav_pct = cvar_config.max_position_cvar_nav_pct
    
    return result


def compute_all_position_cvars(
    positions: Sequence[Mapping[str, Any]],
    nav_usd: float,
    cvar_config: CVaRConfig | None = None,
    ohlcv_dir: Path | str = DEFAULT_OHLCV_DIR,
) -> Dict[str, CVaRResult]:
    """
    Compute CVaR for all positions.
    
    Args:
        positions: List of position dicts
        nav_usd: Total portfolio NAV
        cvar_config: CVaR configuration
        ohlcv_dir: Directory for OHLCV data
        
    Returns:
        Dict mapping symbol to CVaRResult
    """
    if cvar_config is None:
        cvar_config = load_cvar_config()
    
    results = {}
    
    for pos in positions:
        symbol = pos.get("symbol")
        notional = abs(float(pos.get("notional", pos.get("positionAmt", 0)) or 0))
        
        if not symbol or notional <= 0:
            continue
        
        result = compute_position_cvar_for_symbol(
            symbol,
            notional,
            nav_usd,
            cvar_config,
            ohlcv_dir,
        )
        results[symbol] = result
    
    return results


# ---------------------------------------------------------------------------
# Risk Veto Helpers
# ---------------------------------------------------------------------------

def check_portfolio_var_limit(
    var_result: VaRResult,
    var_config: VaRConfig,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if portfolio VaR exceeds limit.
    
    Args:
        var_result: Computed VaR result
        var_config: VaR configuration with limit
        
    Returns:
        Tuple of (should_veto, veto_details)
    """
    if not var_config.enabled:
        return False, {}
    
    if var_result.var_nav_pct > var_config.max_portfolio_var_nav_pct:
        return True, {
            "reason": "portfolio_var_limit",
            "observed": {
                "portfolio_var_nav_pct": var_result.var_nav_pct,
                "portfolio_var_usd": var_result.var_usd,
                "portfolio_volatility": var_result.portfolio_volatility,
            },
            "limits": {
                "max_portfolio_var_nav_pct": var_config.max_portfolio_var_nav_pct,
            },
        }
    
    return False, {}


def check_position_cvar_limit(
    cvar_result: CVaRResult,
    cvar_config: CVaRConfig,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if position CVaR exceeds limit.
    
    Args:
        cvar_result: Computed CVaR result
        cvar_config: CVaR configuration with limit
        
    Returns:
        Tuple of (should_veto, veto_details)
    """
    if not cvar_config.enabled:
        return False, {}
    
    if cvar_result.cvar_nav_pct > cvar_config.max_position_cvar_nav_pct:
        return True, {
            "reason": "position_cvar_limit",
            "observed": {
                "symbol": cvar_result.symbol,
                "position_cvar_nav_pct": cvar_result.cvar_nav_pct,
                "position_cvar_usd": cvar_result.cvar_usd,
                "position_notional_usd": cvar_result.position_notional_usd,
            },
            "limits": {
                "max_position_cvar_nav_pct": cvar_config.max_position_cvar_nav_pct,
            },
        }
    
    return False, {}


# ---------------------------------------------------------------------------
# State Publishing Helper
# ---------------------------------------------------------------------------

def build_risk_advanced_snapshot(
    positions: Sequence[Mapping[str, Any]],
    nav_usd: float,
    var_config: VaRConfig | None = None,
    cvar_config: CVaRConfig | None = None,
) -> Dict[str, Any]:
    """
    Build complete risk snapshot for state publishing.
    
    Returns dict suitable for JSON serialization and dashboard consumption.
    """
    import time
    
    if var_config is None:
        var_config = load_var_config()
    if cvar_config is None:
        cvar_config = load_cvar_config()
    
    snapshot: Dict[str, Any] = {
        "updated_ts": time.time(),
    }
    
    # Compute Portfolio VaR
    if var_config.enabled:
        var_result = compute_portfolio_var_from_positions(positions, nav_usd, var_config)
        snapshot["var"] = {
            "portfolio_var_usd": var_result.var_usd,
            "portfolio_var_nav_pct": var_result.var_nav_pct,
            "portfolio_volatility": var_result.portfolio_volatility,
            "max_portfolio_var_nav_pct": var_config.max_portfolio_var_nav_pct,
            "confidence": var_config.confidence,
            "within_limit": var_result.within_limit,
            "n_assets": var_result.n_assets,
            "lookback_used": var_result.lookback_used,
            "weights": var_result.weights,
        }
    
    # Compute Position CVaRs
    if cvar_config.enabled:
        cvar_results = compute_all_position_cvars(positions, nav_usd, cvar_config)
        snapshot["cvar"] = {
            "per_symbol": {
                symbol: {
                    "cvar_nav_pct": result.cvar_nav_pct,
                    "cvar_usd": result.cvar_usd,
                    "var_nav_pct": result.var_nav_pct,
                    "var_usd": result.var_usd,
                    "limit": cvar_config.max_position_cvar_nav_pct,
                    "within_limit": result.within_limit,
                    "position_notional_usd": result.position_notional_usd,
                    "lookback_used": result.lookback_used,
                }
                for symbol, result in cvar_results.items()
            },
            "max_position_cvar_nav_pct": cvar_config.max_position_cvar_nav_pct,
            "confidence": cvar_config.confidence,
        }
    
    return snapshot
