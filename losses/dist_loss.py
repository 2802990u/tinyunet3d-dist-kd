"""DIST (Pearson-correlation) distillation loss used in the paper.

Faithful standalone copy of ``dist_kd`` from ``brats/train_student_frac_brats.py`` and
``msd/train_student_msd.py`` (multi-class, softmax head).  Given student/teacher logits
``(B, C, D, H, W)`` and the ground-truth label ``(B, D, H, W)``:

* inter-class term : 1 - Pearson correlation across the C classes, per voxel
                     (does the student rank the classes like the teacher?)
* intra-class term : 1 - Pearson correlation across the **foreground** voxels, per class
                     (does the student's spatial pattern of each class match the teacher's?)

    L_DIST = beta * inter + gamma * intra          (beta = gamma = 1, T = 1 in all runs)
    L      = DiceCE(student, y) + lambda * L_DIST  (lambda = 1)

The foreground mask on the intra term removes the background-dominated noise that made the
unmasked variant unstable.  ``eps_in_sqrt=True`` is the numerically safer form used in
``spleen/spleen_student.py`` (``eps`` inside the square root avoids NaN gradients when a
variance is exactly zero); the default (False) matches the BraTS / MSD training scripts.
"""
import torch
import torch.nn.functional as F


def pearson(a, b, dim, eps=1e-6, eps_in_sqrt=False):
    a = a - a.mean(dim=dim, keepdim=True)
    b = b - b.mean(dim=dim, keepdim=True)
    if eps_in_sqrt:
        den = (a.pow(2).sum(dim=dim) + eps).sqrt() * (b.pow(2).sum(dim=dim) + eps).sqrt()
    else:
        den = a.pow(2).sum(dim=dim).sqrt() * b.pow(2).sum(dim=dim).sqrt() + eps
    return (a * b).sum(dim=dim) / den


def dist_loss(s_logits, t_logits, label, T=1.0, beta=1.0, gamma=1.0, eps_in_sqrt=False):
    """Foreground-masked DIST. s_logits, t_logits: (B, C, ...); label: (B, ...) integer mask."""
    B, C = s_logits.shape[0], s_logits.shape[1]
    s = F.softmax(s_logits.reshape(B, C, -1) / T, dim=1)       # (B, C, N)
    t = F.softmax(t_logits.reshape(B, C, -1) / T, dim=1)
    inter = (1.0 - pearson(s, t, dim=1, eps_in_sqrt=eps_in_sqrt)).mean()
    fg = label.reshape(B, -1) > 0
    terms = []
    for b in range(B):
        m = fg[b]
        if m.sum() < 2:
            continue
        corr = pearson(s[b][:, m], t[b][:, m], dim=1, eps_in_sqrt=eps_in_sqrt)   # (C,)
        terms.append((1.0 - corr).mean())
    intra = torch.stack(terms).mean() if terms else inter.new_zeros(())
    return beta * inter + gamma * intra


if __name__ == "__main__":
    torch.manual_seed(0)
    s = torch.randn(2, 4, 16, 16, 16, requires_grad=True)
    t = torch.randn(2, 4, 16, 16, 16)
    y = (torch.rand(2, 16, 16, 16) > 0.7).long()
    loss = dist_loss(s, t, y)
    loss.backward()
    print("random vs random :", float(loss), " grad finite:", bool(torch.isfinite(s.grad).all()))
    print("identical logits :", float(dist_loss(t, t, y)), "(should be ~0)")
