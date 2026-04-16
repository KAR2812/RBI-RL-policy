import numpy as np
from typing import Optional, List, Dict, Any

def compute_metrics(
    trajectories: List[Dict[str, Any]],
    pi_target: float = 0.04,
    w_y: float = 1.0,
    w_i: float = 0.5,
) -> dict:
    ep_returns = []
    ep_macro_losses = []
    ep_inf_losses = []
    ep_unemp_losses = []
    ep_smooth_losses = []
    
    all_pi = []
    all_y = []
    all_di = []
    
    recoveries = []

    for ep in trajectories:
        pi = np.asarray(ep["inflation"], dtype=float)
        y = np.asarray(ep["output_gap"], dtype=float)
        i = np.asarray(ep["interest_rate"], dtype=float)
        r = np.asarray(ep["reward"], dtype=float)
        shock = np.asarray(ep["shock_active"], dtype=bool)

        pi_dev = pi - pi_target
        di = np.diff(i, prepend=i[0] if len(i) > 0 else 0.0) 
        
        inf_loss = pi_dev**2
        unemp_loss = w_y * (y**2)
        smooth_loss = w_i * (di**2)
        macro_loss = inf_loss + unemp_loss + smooth_loss

        ep_returns.append(np.sum(r))
        ep_macro_losses.append(np.sum(macro_loss))
        ep_inf_losses.append(np.sum(inf_loss))
        ep_unemp_losses.append(np.sum(unemp_loss))
        ep_smooth_losses.append(np.sum(smooth_loss))
        
        all_pi.extend(pi)
        all_y.extend(y)
        all_di.extend(di)
        
        # Recovery time
        rec = _compute_recovery_time(pi, pi_target, shock)
        if rec >= 0:
            recoveries.append(rec)

    metrics = {
        "Total Macro Loss": float(np.mean(ep_macro_losses)),
        "Inflation Loss": float(np.mean(ep_inf_losses)),
        "Unemployment Loss": float(np.mean(ep_unemp_losses)),
        "Interest Rate Smoothness": float(np.mean(ep_smooth_losses)),
        
        "Mean Return": float(np.mean(ep_returns)),
        "Variance of Returns": float(np.var(ep_returns)),
        
        "Inflation Variance": float(np.var(all_pi)),
        "Output Gap Variance": float(np.var(all_y)),
        
        "Recovery Time After Shock": float(np.mean(recoveries)) if recoveries else -1.0,
        "Policy Reactivity": float(np.std(all_di)), 
    }
    
    return metrics

def _compute_recovery_time(
    pi: np.ndarray,
    pi_target: float,
    shock_flags: np.ndarray,
    threshold: float = 0.005,
) -> int:
    """
    Count steps from shock-end until |pi - pi*| < threshold.
    Returns -1 if recovery never happens.
    """
    if not np.any(shock_flags):
        return 0

    shock_end_idx = int(np.where(shock_flags)[0][-1])
    for idx in range(shock_end_idx, len(pi)):
        if abs(pi[idx] - pi_target) < threshold:
            return int(idx - shock_end_idx)

    return -1   # never fully recovered
