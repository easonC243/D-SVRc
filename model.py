import numpy as np

def dynamic_penalty(label, diff, base_c_C, base_star_c_C, base_c_ε, base_star_c_ε, min_label, max_label, max_abs_diff):
    weight_factor = (label - min_label) / (max_label - min_label + 1e-8)
    normalized_diff = abs(diff) / (max_abs_diff + 1e-8)
    penalty_factor = 0.6 + 1.5 / (1 + np.exp(-10 * (normalized_diff - 0.5)))

    c_C = np.clip(base_c_C * (1 - weight_factor) * penalty_factor, 1e-2, 10)
    star_c_C = np.clip(base_star_c_C * (1 - weight_factor) * penalty_factor, 1e-2, 10)
    c_ε = np.clip(base_c_ε * (weight_factor + 0.1) / penalty_factor, 1e-3, 0.5)
    star_c_ε = np.clip(base_star_c_ε * (weight_factor + 0.1) / penalty_factor, 1e-3, 0.5)

    return c_C, star_c_C, c_ε, star_c_ε


def svr_loss_function(X, y, w, b, base_c_C, base_star_c_C, base_c_ε, base_star_c_ε, censored, epsilon, C):
    n = len(y)
    reg_term = 0.5 * np.dot(w, w)
    total_loss = 0

    min_label, max_label = np.min(y), np.max(y)
    all_diff = y - (X @ w + b)
    max_abs_diff = np.max(np.abs(all_diff))

    for i in range(n):
        diff = all_diff[i]
        c_C, star_c_C, c_ε, star_c_ε = dynamic_penalty(
            y[i], diff, base_c_C, base_star_c_C, base_c_ε, base_star_c_ε, min_label, max_label, max_abs_diff
        )

        if censored[i]: 
            if diff > c_ε:
                total_loss += c_C * (abs(diff) - c_ε)
            elif diff < -star_c_ε:
                total_loss += star_c_C * (abs(diff) - star_c_ε)
        else: 
            if abs(diff) > epsilon:
                total_loss += C * (abs(diff) - epsilon)

    return reg_term + total_loss


def train_svr(X, y, w, b, C, epsilon, base_c_C, base_star_c_C, base_c_ε, base_star_c_ε, censored, epochs, learning_rate):
    n, d = X.shape
    min_label, max_label = np.min(y), np.max(y)
    loss_history = []

    for epoch in range(epochs):
        preds = X @ w + b
        diffs = y - preds
        max_abs_diff = np.max(np.abs(diffs))
        loss = svr_loss_function(X, y, w, b, base_c_C, base_star_c_C, base_c_ε, base_star_c_ε, censored, epsilon, C)
        loss_history.append(loss)

        grad_w = np.zeros_like(w)
        grad_b = 0.0

        for i in range(n):
            diff = diffs[i]
            xi = X[i]

            c_C, star_c_C, c_ε, star_c_ε = dynamic_penalty(
                y[i], diff, base_c_C, base_star_c_C, base_c_ε, base_star_c_ε,
                min_label, max_label, max_abs_diff
            )

            if censored[i]:
                if diff > c_ε:
                    grad_w -= c_C * xi
                    grad_b -= c_C
                elif diff < -star_c_ε:
                    grad_w += star_c_C * xi
                    grad_b += star_c_C
            else:
                if diff > epsilon:
                    grad_w -= C * xi
                    grad_b -= C
                elif diff < -epsilon:
                    grad_w += C * xi
                    grad_b += C

        grad_w += w

        w -= learning_rate * grad_w / n
        b -= learning_rate * grad_b / n

    return w, b
