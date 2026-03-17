"""
Viterbi post-processing for anatomy sequence predictions.

VCE (Video Capsule Endoscopy) anatomy progression order:
  mouth(0) → esophagus(1) → z-line(5) → stomach(2) → pylorus(6)
  → small intestine(3) → ileocecal valve(7) → colon(4)

Viterbi enforces this biological sequence constraint on per-frame predictions,
boosting short-duration transition classes (pylorus, z-line, ileocecal valve).
"""

import numpy as np

# Linear sequence index in ANATOMY_LABELS order:
# mouth=0, esophagus=1, stomach=2, SI=3, colon=4, z-line=5, pylorus=6, ileocecal=7
_ANAT_LINEAR_SEQ = [0, 1, 5, 2, 6, 3, 7, 4]
_POS_IN_SEQ = {cls: i for i, cls in enumerate(_ANAT_LINEAR_SEQ)}

# Log transition matrix (precomputed)
_N_ANAT = 8
_LOG_TRANS = None


def _build_log_trans():
    global _LOG_TRANS
    if _LOG_TRANS is not None:
        return _LOG_TRANS

    LOG_STAY  = np.log(0.90)
    LOG_ADJ   = np.log(0.75)   # 1 step apart (e.g., stomach → pylorus)
    LOG_SKIP1 = np.log(0.15)   # 2 steps (e.g., stomach → SI, skipping pylorus)
    LOG_SKIP2 = np.log(0.03)   # 3 steps
    LOG_FAR   = np.log(1e-4)   # physiologically impossible

    t = np.full((_N_ANAT, _N_ANAT), LOG_FAR, dtype=np.float64)
    for i in range(_N_ANAT):
        for j in range(_N_ANAT):
            if i == j:
                t[i, j] = LOG_STAY
            else:
                dist = abs(_POS_IN_SEQ[i] - _POS_IN_SEQ[j])
                if dist == 1:
                    t[i, j] = LOG_ADJ
                elif dist == 2:
                    t[i, j] = LOG_SKIP1
                elif dist == 3:
                    t[i, j] = LOG_SKIP2
    _LOG_TRANS = t
    return _LOG_TRANS


def viterbi_anatomy(pred_probs_anatomy: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """
    Apply Viterbi smoothing to anatomy sigmoid probabilities.

    Args:
        pred_probs_anatomy: [T, 8] sigmoid probabilities
        alpha: blend weight — alpha fraction of output is Viterbi-guided,
               (1-alpha) fraction retains original distribution.
               alpha=1.0 → hard Viterbi  alpha=0.0 → no change

    Returns:
        [T, 8] smoothed probabilities (same shape, values in [0,1])
    """
    T, N = pred_probs_anatomy.shape
    if T == 0 or N != _N_ANAT:
        return pred_probs_anatomy

    log_trans = _build_log_trans()

    # Normalize to probability distribution for emission
    probs = pred_probs_anatomy.astype(np.float64)
    probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)
    log_emit = np.log(probs + 1e-8)  # [T, 8]

    # Viterbi forward pass
    V       = np.full((T, N), -np.inf, dtype=np.float64)
    backptr = np.zeros((T, N), dtype=np.int32)
    V[0]    = log_emit[0]

    for t in range(1, T):
        trans_scores = V[t - 1, :, None] + log_trans  # [N, N]: from i to j
        best         = trans_scores.argmax(axis=0)     # [N]: best predecessor for each j
        V[t]         = trans_scores[best, np.arange(N)] + log_emit[t]
        backptr[t]   = best

    # Backtrack
    path     = np.zeros(T, dtype=np.int32)
    path[-1] = int(V[-1].argmax())
    for t in range(T - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]

    # Soft blend: winner keeps its original score boosted by alpha,
    # non-winners are suppressed to (1-alpha) * original
    smoothed = pred_probs_anatomy * (1.0 - alpha)
    for t in range(T):
        smoothed[t, path[t]] += alpha * pred_probs_anatomy[t, path[t]]

    return smoothed.astype(np.float32)
