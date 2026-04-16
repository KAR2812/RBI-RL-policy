import numpy as np

class TaylorRule:
    """
    Standard Taylor Rule baseline.
    i_t = r* + pi_t + alpha*(pi_t - pi*) + beta*y_t
    """
    def __init__(self, r_star=0.02, alpha=0.5, beta=0.5, pi_star=0.04):
        self.r_star = r_star
        self.alpha = alpha
        self.beta = beta
        self.pi_star = pi_star

    def predict(self, obs, deterministic=True):
        # obs: [pi_t, y_t, i_t, E_t[pi_{t+1}]]
        pi_t = obs[0]
        y_t = obs[1]
        
        i_t = self.r_star + pi_t + self.alpha * (pi_t - self.pi_star) + self.beta * y_t
        
        # Clip to [0.0, 0.20]
        i_t_clipped = np.clip(i_t, 0.0, 0.20)
        return np.array([i_t_clipped], dtype=np.float32), None

class AggressiveTaylorRule(TaylorRule):
    """
    Aggressive Taylor Rule with higher inflation responsiveness.
    """
    def __init__(self, r_star=0.02, alpha=1.5, beta=0.5, pi_star=0.04):
        super().__init__(r_star=r_star, alpha=alpha, beta=beta, pi_star=pi_star)

class FixedInflationTargeting:
    """
    Fixed Inflation Targeting baseline.
    always set i = 0.06 (r* + pi*)
    """
    def __init__(self, target_rate=0.06):
        self.target_rate = target_rate

    def predict(self, obs, deterministic=True):
        # Return constant rate rule mapped to [0.0, 0.20] clip
        i_t_clipped = np.clip(self.target_rate, 0.0, 0.20)
        return np.array([i_t_clipped], dtype=np.float32), None
