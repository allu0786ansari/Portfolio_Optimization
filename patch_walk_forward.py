"""Fixes the policy.actor AttributeError in backtesting/walk_forward.py.

PPO uses ActorCriticPolicy — action is obtained via policy._predict()
SAC uses SACPolicy     — action is obtained via policy.actor()

This patch replaces run_rl_on_window to handle both correctly.
Run from project root: python patch_walk_forward.py
"""
from pathlib import Path

path = Path("backtesting/walk_forward.py")
content = path.read_text(encoding="utf-8")

OLD = '''def run_rl_on_window(
    policy,
    feature_matrix: np.ndarray,
    return_matrix: np.ndarray,
    start_idx: int,
    end_idx: int,
    n_assets: int,
) -> np.ndarray:
    """Run RL policy deterministically on a test window.

    Reconstructs observations manually (no Gym overhead) for speed.
    Returns array of daily portfolio log returns.
    """
    weights = np.ones(n_assets, dtype=np.float32) / n_assets
    returns = []

    for t in range(start_idx, end_idx):
        # Build observation: features + current weights
        features_flat = feature_matrix[t].flatten().astype(np.float32)
        features_flat = np.clip(features_flat, -10.0, 10.0)
        obs = np.concatenate([features_flat, weights])

        # Get action from policy (deterministic)
        import torch
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action = policy.actor(obs_tensor).squeeze(0).numpy()

        weights    = softmax(action)
        port_ret   = float(np.dot(weights, return_matrix[t]))
        returns.append(port_ret)

    return np.array(returns)'''

NEW = '''def run_rl_on_window(
    policy,
    feature_matrix: np.ndarray,
    return_matrix: np.ndarray,
    start_idx: int,
    end_idx: int,
    n_assets: int,
) -> np.ndarray:
    """Run RL policy deterministically on a test window.

    Handles both PPO (ActorCriticPolicy) and SAC (SACPolicy).
    PPO: uses policy._predict() — the unified SB3 interface.
    SAC: has policy.actor attribute directly.
    Returns array of daily portfolio log returns.
    """
    import torch

    weights = np.ones(n_assets, dtype=np.float32) / n_assets
    returns = []

    # Detect policy type once
    has_actor = hasattr(policy, "actor")

    for t in range(start_idx, end_idx):
        features_flat = feature_matrix[t].flatten().astype(np.float32)
        features_flat = np.clip(features_flat, -10.0, 10.0)
        obs = np.concatenate([features_flat, weights])

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            if has_actor:
                # SAC policy
                action = policy.actor(obs_tensor).squeeze(0).numpy()
            else:
                # PPO ActorCriticPolicy — use _predict (deterministic=True)
                action_tensor, _ = policy._predict(obs_tensor, deterministic=True)
                action = action_tensor.squeeze(0).numpy()

        weights  = softmax(action)
        port_ret = float(np.dot(weights, return_matrix[t]))
        returns.append(port_ret)

    return np.array(returns)'''

if OLD in content:
    content = content.replace(OLD, NEW)
    path.write_text(content, encoding="utf-8")
    print("Patched: run_rl_on_window now handles both PPO and SAC")
else:
    # Fallback: try to find and replace just the actor line
    old_line = "            action = policy.actor(obs_tensor).squeeze(0).numpy()"
    new_lines = (
        "            if hasattr(policy, 'actor'):\n"
        "                action = policy.actor(obs_tensor).squeeze(0).numpy()\n"
        "            else:\n"
        "                action_tensor, _ = policy._predict(obs_tensor, deterministic=True)\n"
        "                action = action_tensor.squeeze(0).numpy()"
    )
    if old_line in content:
        content = content.replace(old_line, new_lines)
        path.write_text(content, encoding="utf-8")
        print("Patched: replaced actor line with PPO/SAC-compatible version")
    else:
        print("Could not find target line — check walk_forward.py manually")
        print("Find this line:")
        print("    action = policy.actor(obs_tensor).squeeze(0).numpy()")
        print("Replace with:")
        print("    if hasattr(policy, 'actor'):")
        print("        action = policy.actor(obs_tensor).squeeze(0).numpy()")
        print("    else:")
        print("        action_tensor, _ = policy._predict(obs_tensor, deterministic=True)")
        print("        action = action_tensor.squeeze(0).numpy()")
        raise SystemExit(1)

print()
print("Now run: python -m backtesting.walk_forward")