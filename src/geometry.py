import numpy as np
from scipy.optimize import minimize_scalar


def pca(vecs, n_components=None):
    """Standard PCA via SVD. Returns (scores, variances, loadings)."""
    X = vecs - vecs.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    if n_components is not None:
        U, S, Vt = U[:, :n_components], S[:n_components], Vt[:n_components]
    scores = U * S
    variances = S**2 / max(len(vecs) - 1, 1)
    return scores, variances, Vt.T


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-14 else 0.0


# ── Periodic BC: ellipse fitting ──────────────────────────────────────────────

def _fit_ellipse_pair(xy, u, n):
    """Fit a cos/sin ellipse at harmonic *n* to 2D points with phases *u* in [0,1).

    Model per axis:  x_i = a*cos(2πnu_i) + b*sin(2πnu_i) + c
    Returns (x_coeffs, y_coeffs) each of shape (3,): [a, b, c].
    """
    theta = 2 * np.pi * n * u
    A = np.column_stack([np.cos(theta), np.sin(theta), np.ones_like(theta)])
    x_coeffs, _, _, _ = np.linalg.lstsq(A, xy[:, 0], rcond=None)
    y_coeffs, _, _, _ = np.linalg.lstsq(A, xy[:, 1], rcond=None)
    return x_coeffs, y_coeffs


def _fit_ellipse_robust(xy, u, words, n, sigma_clip=2.0):
    """Robust ellipse fit with sigma-clipping. Returns dict with coefficients
    and reported R² (geometric mean of per-axis R²)."""
    N = len(words)
    mask = np.ones(N, dtype=bool)
    min_inliers = max(4, int(np.ceil(0.6 * N)))

    if sigma_clip is not None:
        for _ in range(3):
            xc, yc = _fit_ellipse_pair(xy[mask], u[mask], n)
            theta = 2 * np.pi * n * u
            A = np.column_stack([np.cos(theta), np.sin(theta), np.ones_like(theta)])
            x_pred = A @ xc
            y_pred = A @ yc
            residuals = np.sqrt((xy[:, 0] - x_pred)**2 + (xy[:, 1] - y_pred)**2)
            mad = np.median(residuals[mask])
            new_mask = residuals <= sigma_clip * (mad + 1e-12)
            if new_mask.sum() < min_inliers:
                idx = np.argsort(residuals)
                new_mask = np.zeros(N, dtype=bool)
                new_mask[idx[:min_inliers]] = True
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask

    xc, yc = _fit_ellipse_pair(xy, u, n)
    theta = 2 * np.pi * n * u
    A = np.column_stack([np.cos(theta), np.sin(theta), np.ones_like(theta)])
    r2_x = r2(xy[:, 0], A @ xc)
    r2_y = r2(xy[:, 1], A @ yc)
    r2_reported = np.sqrt(max(0, r2_x) * max(0, r2_y))

    return {
        "x_coeffs":      xc.tolist(),
        "y_coeffs":      yc.tolist(),
        "r2_x":          r2_x,
        "r2_y":          r2_y,
        "r2_reported":   r2_reported,
        "inlier_mask":   mask.tolist(),
        "outlier_words": [w for i, w in enumerate(words) if not mask[i]],
        "n_inliers":     int(mask.sum()),
    }


def best_ellipse_in_subspace(coords, phases, words, n_harmonics=2,
                              max_pairs=None, sigma_clip=2.0):
    """Search PC pairs for the best ellipse fit across harmonic modes.

    *phases* should be in [0, 1) representing the cyclic position.
    """
    k = coords.shape[1]
    u = np.asarray(phases, dtype=float)
    best_r2, best_result = -np.inf, None
    pair_results = []
    n_checked = 0

    for i in range(k):
        for j in range(i + 1, k):
            if max_pairs is not None and n_checked >= max_pairs:
                break
            for n in range(1, n_harmonics + 1):
                res = _fit_ellipse_robust(coords[:, [i, j]], u, words, n, sigma_clip)
                res["pc_pair"] = (i, j)
                res["harmonic"] = n
                pair_results.append(res)
                if res["r2_reported"] > best_r2:
                    best_r2, best_result = res["r2_reported"], res
            n_checked += 1

    return {
        "best_r2":       best_r2,
        "best_pair":     best_result["pc_pair"] if best_result else None,
        "best_harmonic": best_result["harmonic"] if best_result else None,
        "best_fit":      best_result,
        "all_pairs":     pair_results,
    }


# ── Neumann: rotation-agnostic parabola fitting ──────────────────────────────

def _fit_parabola_at_angle(xy, angle):
    c, s = np.cos(angle), np.sin(angle)
    rot = xy @ np.array([[c, -s], [s, c]]).T
    x, y = rot[:, 0], rot[:, 1]
    coeffs = np.polyfit(x, y, 2)
    return coeffs, r2(y, np.polyval(coeffs, x)), rot


def _best_rotation(xy, n_coarse=180):
    """Coarse grid search + golden-section refinement for the angle that
    maximises parabola R² on the 2D point cloud."""
    angles = np.linspace(0, np.pi, n_coarse, endpoint=False)
    r2s = [_fit_parabola_at_angle(xy, a)[1] for a in angles]
    best = angles[int(np.argmax(r2s))]

    lo, hi = best - np.pi / n_coarse * 3, best + np.pi / n_coarse * 3
    try:
        res = minimize_scalar(lambda a: -_fit_parabola_at_angle(xy, a)[1],
                              bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-4})
        best = res.x
    except Exception:
        pass

    coeffs, best_r2, rot = _fit_parabola_at_angle(xy, best)
    return {
        "r2": best_r2,
        "angle_deg": float(np.degrees(best) % 180),
        "coeffs": coeffs.tolist(),
        "x_rot": rot[:, 0].tolist(),
        "y_rot": rot[:, 1].tolist(),
        "y_fit": np.polyval(coeffs, rot[:, 0]).tolist(),
    }


def _fit_parabola_robust(xy, words, sigma_clip=2.0, n_angles=180):
    """Rotation-agnostic parabola fit with optional sigma-clipping.
    Pass sigma_clip=None to disable clipping and use all points.
    The reported R² is always on the full data."""
    N = len(words)
    mask = np.ones(N, dtype=bool)
    min_inliers = max(4, int(np.ceil(0.6 * N)))

    if sigma_clip is not None:
        for _ in range(3):
            sub = _best_rotation(xy[mask], n_angles)
            angle_rad = np.radians(sub["angle_deg"])
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            rot_all = xy @ np.array([[c, -s], [s, c]]).T
            residuals = np.abs(rot_all[:, 1] - np.polyval(sub["coeffs"], rot_all[:, 0]))
            mad = np.median(residuals[mask])
            new_mask = residuals <= sigma_clip * (mad + 1e-12)
            if new_mask.sum() < min_inliers:
                idx = np.argsort(residuals)
                new_mask = np.zeros(N, dtype=bool)
                new_mask[idx[:min_inliers]] = True
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask

    full = _best_rotation(xy, n_angles)          
    inlier = _best_rotation(xy[mask], n_angles)
    return {
        "r2_all_points": full["r2"],
        "r2_inliers":    inlier["r2"],
        "r2_reported":   full["r2"],
        "angle_deg":     full["angle_deg"],
        "coeffs":        full["coeffs"],
        "x_rot":         full["x_rot"],
        "y_rot":         full["y_rot"],
        "y_fit":         full["y_fit"],
        "inlier_mask":   mask.tolist(),
        "outlier_words": [w for i, w in enumerate(words) if not mask[i]],
        "n_inliers":     int(mask.sum()),
    }


def best_parabola_in_subspace(coords, words, max_pairs=None, sigma_clip=2.0,
                               variances=None):
    """Search PC pairs for the best parabola fit.

    *max_pairs*=None means exhaustive search over all C(k,2) combinations.
    If *variances* (from ``pca``) are provided, each result includes
    ``var_fraction`` — the share of total PCA variance captured by
    the two PCs in that pair.
    """
    k = coords.shape[1]
    best_r2, best_result = -np.inf, None
    pair_results = []
    n_checked = 0

    total_var = float(variances.sum()) if variances is not None else None

    for i in range(k):
        for j in range(i + 1, k):
            if max_pairs is not None and n_checked >= max_pairs:
                break
            res = _fit_parabola_robust(coords[:, [i, j]], words, sigma_clip)
            res["pc_pair"] = (i, j)
            if variances is not None:
                res["var_fraction"] = float((variances[i] + variances[j]) / total_var)
            pair_results.append(res)
            if res["r2_reported"] > best_r2:
                best_r2, best_result = res["r2_reported"], res
            n_checked += 1

    return {
        "best_r2":        best_r2,
        "best_pair":      best_result["pc_pair"] if best_result else None,
        "best_fit":       best_result,
        "all_pairs":      pair_results,
        "n_pairs_tried":  n_checked,
    }


# ── Log scale: chirp mode detection ──────────────────────────────────────────

def best_chirp_in_subspace(coords, log_coords, words, n_modes=3, sigma_clip=2.0):
    """
    Find the linear combination of PCs that best matches sin(n*π/2 * u)
    for u = normalised log-position, n = 1, 2, ..., n_modes.
    """
    N = coords.shape[0]
    k = max(2, min(coords.shape[1], N // 2))
    coords = coords[:, :k]
    u = (log_coords - log_coords.min()) / (log_coords.max() - log_coords.min() + 1e-14)
    coords_aug = np.hstack([coords, np.ones((N, 1))])

    results = {}
    for n in range(1, n_modes + 1):
        target = np.sin(n * np.pi / 2 * u)
        mask = np.ones(N, dtype=bool)

        for _ in range(3):
            w, _, _, _ = np.linalg.lstsq(coords_aug[mask], target[mask], rcond=None)
            proj = coords_aug @ w
            residuals = np.abs(target - proj)
            mad = np.median(residuals[mask])
            new_mask = residuals <= sigma_clip * (mad + 1e-12)
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask

        r2_all = r2(target, proj)
        results[f"mode_{n}"] = {
            "r2_all":        r2_all,
            "r2_inliers":    r2(target[mask], proj[mask]),
            "projection":    proj.tolist(),
            "target":        target.tolist(),
            "pc_weights":    w[:-1].tolist(),
            "intercept":     float(w[-1]),
            "outlier_words": [words[i] for i in range(N) if not mask[i]],
            "n_inliers":     int(mask.sum()),
        }

    best_mode = max(results, key=lambda k: results[k]["r2_all"])
    return {
        "best_mode":         best_mode,
        "best_r2":           results[best_mode]["r2_all"],
        "best_r2_inliers":   results[best_mode]["r2_inliers"],
        "modes":             results,
    }


# ── 2-D Neumann BC: mode fitting on a rectangular domain ─────────────────────

def neumann_2d_mode(u, v, m, n):
    """Evaluate the (m, n) Neumann eigenmode cos(m*pi*u) * cos(n*pi*v)."""
    return np.cos(m * np.pi * u) * np.cos(n * np.pi * v)


def fit_2d_modes(coords, u, v, max_m=3, max_n=3):
    """Fit each PC to the best-matching 2-D Neumann mode.

    For each column of *coords* (i.e. each PC), find the (m, n) mode
    that maximises R² via linear regression.

    Returns a list of dicts, one per PC, sorted by the PC index.
    """
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    k = coords.shape[1]
    results = []

    for pc_idx in range(k):
        pc = coords[:, pc_idx]
        best_r2, best_mn = -np.inf, (0, 0)
        best_info = {}
        for m in range(max_m + 1):
            for n in range(max_n + 1):
                if m == 0 and n == 0:
                    continue
                mode = neumann_2d_mode(u, v, m, n)
                A = np.column_stack([mode, np.ones(len(u))])
                w, _, _, _ = np.linalg.lstsq(A, pc, rcond=None)
                pred = A @ w
                score = r2(pc, pred)
                if score > best_r2:
                    best_r2 = score
                    best_mn = (m, n)
                    best_info = {
                        "alpha": float(w[0]),
                        "beta": float(w[1]),
                        "pred": pred,
                    }

        results.append({
            "pc":       pc_idx,
            "mode":     best_mn,
            "r2":       best_r2,
            "alpha":    best_info["alpha"],
            "beta":     best_info["beta"],
            "pred":     best_info["pred"].tolist(),
        })

    return results


def fit_2d_modes_multipc(coords, u, v, max_m=3, max_n=3):
    """For each 2-D Neumann mode, find the linear combination of PCs
    that best reproduces it (via least-squares).

    This is the dual of ``fit_2d_modes``: instead of asking which mode
    best matches each PC, we ask which PC mixture best matches each mode.

    Returns a dict keyed by (m, n) tuples.
    """
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    N, k = coords.shape
    coords_aug = np.hstack([coords, np.ones((N, 1))])

    results = {}
    for m in range(max_m + 1):
        for n in range(max_n + 1):
            if m == 0 and n == 0:
                continue
            target = neumann_2d_mode(u, v, m, n)
            w, _, _, _ = np.linalg.lstsq(coords_aug, target, rcond=None)
            pred = coords_aug @ w
            score = r2(target, pred)
            results[(m, n)] = {
                "r2":         score,
                "pc_weights": w[:-1].tolist(),
                "intercept":  float(w[-1]),
                "pred":       pred.tolist(),
                "target":     target.tolist(),
            }

    return results
