"""Correct-answer-loss calibration and deterministic answer-preserving routing."""
import numpy as np


def select_threshold(train_scores, calibration_scores, standard, correct, budget):
    """Maximize the threshold subject to empirical gross loss on calibration.

    The denominator is all Standard calibration examples. A tied score is kept.
    Evaluation labels and evaluation scores are not inputs to threshold selection.
    """
    train_scores = np.asarray(train_scores, dtype=float)
    calibration_scores = np.asarray(calibration_scores, dtype=float)
    standard, correct = np.asarray(standard, dtype=bool), np.asarray(correct, dtype=bool)
    if not 0 <= budget <= 1 or not standard.any():
        raise ValueError('A budget in [0, 1] and nonempty Standard calibration set are required.')
    if not np.isfinite(train_scores).all() or not np.isfinite(calibration_scores).all():
        raise ValueError('Probe scores must be finite.')
    grid = np.r_[-np.inf, np.unique(train_scores), np.inf]
    protected_scores = np.sort(calibration_scores[standard & correct])
    lost = np.searchsorted(protected_scores, grid, side='left')
    index = np.flatnonzero(lost / standard.sum() <= budget + 1e-12)[-1]
    return float(grid[index]), int(lost[index])


def route(rows, answers, scores, threshold):
    """Keep the original response or return that example's displayed NOTA letter."""
    if len(rows) != len(answers) or len(rows) != len(scores):
        raise ValueError('Rows, answers and scores must have equal lengths.')
    output = []
    for row, answer, value in zip(rows, answers, scores):
        if row['sid'] != answer['sid']:
            raise ValueError('Answer IDs must match the manifest order.')
        keep = value >= threshold
        output.append(dict(sid=row['sid'], response=answer['response'] if keep else row['nota'],
                           route='answer' if keep else 'nota'))
    return output
