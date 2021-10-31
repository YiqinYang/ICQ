import torch as th


def build_target_q(td_q, target_q, mac, mask, gamma, td_lambda, n):
    aug = th.zeros_like(td_q[:, :1])
    mac = mac[:, :-1] 
    tree_q_vals = th.zeros_like(td_q)
    coeff = 1.0
    t1 = td_q[:]
    for _ in range(n):
        tree_q_vals += t1 * coeff
        t1 = th.cat(((t1 * mac)[:, 1:], aug), dim=1)
        coeff *= gamma * td_lambda
    return target_q + tree_q_vals
