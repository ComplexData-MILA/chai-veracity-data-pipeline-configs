"""Kleinberg burst detection algorithm for event time series.

Based on: J. Kleinberg, "Bursty and Hierarchical Structure in Streams"
(Data Mining and Knowledge Discovery, 2003).

Given inter-arrival gaps between events, the algorithm assigns each gap a
"burst state" q ∈ {0, 1, 2, ...} where 0 is basal and q ≥ 1 are increasing
burst intensities. The cost model balances fidelity to an exponential
arrival model against a penalty for transitioning to a higher state.

Transition cost:  τ(i, j) = γ × (j - i)  for j > i, 0 otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class BurstSegment:
    """A contiguous burst interval."""
    start_idx: int       # index into gaps array (first gap in burst)
    end_idx: int         # index into gaps array (last gap in burst, inclusive)
    max_state: int       # highest state level reached in this burst
    total_time_s: float  # time spanned by the burst (from first to last event in the segment)
    energy: float        # max_state × total_time_s


@dataclass
class BurstResult:
    """Full burst detection result for a single entity's event sequence."""
    gaps: list[float]      # inter-arrival gaps in seconds
    states: list[int]      # burst state for each gap (0 = basal)
    bursts: list[BurstSegment]
    base_rate: float       # estimated basal event rate (events/sec)
    max_burst_weight: int  # highest state level reached
    total_burst_energy: float  # sum of (burst weight × duration) over all bursts
    weighted_burst_count: float  # number of bursts weighted by max state level


def kleinberg_burst(
    gaps: Sequence[float],
    s: float = 2.0,
    gamma: float = 1.0,
    base_rate: float | None = None,
    max_states: int = 10,
) -> list[int]:
    """Assign a burst state to each inter-arrival gap.

    Args:
        gaps: Inter-arrival times in seconds. Must have length ≥ 2.
        s: Rate scaling factor per burst level (default 2.0).
        gamma: Transition cost coefficient (default 1.0).
        base_rate: Basal event rate in events/sec. Computed from gaps if None.
        max_states: Maximum number of burst states to consider (default 10).

    Returns:
        List of integer states, one per gap.
    """
    n = len(gaps)
    if n < 2:
        # Need at least 2 gaps for meaningful burst detection
        return [0] * n

    if base_rate is None:
        total_time = sum(gaps)
        base_rate = n / total_time if total_time > 0 else 1.0

    K = max_states  # number of states: 0, 1, ..., K-1

    # Precompute rate and cost for each state
    lambdas = [base_rate * (s ** q) for q in range(K)]

    # C[q][i] = cost of being in state q at gap i
    C: list[list[float]] = []
    for q in range(K):
        lam = lambdas[q]
        log_lam = math.log(lam)
        C.append([-log_lam + lam * g for g in gaps])

    # DP: dp[q] = min cost to end in state q at current position
    dp = [C[q][0] for q in range(K)]
    backpointer: list[dict[int, int]] = []  # backpointer[i][q] = previous state

    for i in range(1, n):
        new_dp = [float("inf")] * K
        bp_row: dict[int, int] = {}
        for q in range(K):  # current state
            best_cost = float("inf")
            best_prev = 0
            for q_prev in range(K):
                # Transition cost: only pay to go UP in burst level
                trans = gamma * (q - q_prev) if q > q_prev else 0.0
                cost = dp[q_prev] + trans + C[q][i]
                if cost < best_cost:
                    best_cost = cost
                    best_prev = q_prev
            new_dp[q] = best_cost
            bp_row[q] = best_prev
        dp = new_dp
        backpointer.append(bp_row)

    # Backtrack
    best_q = min(range(K), key=lambda q: dp[q])
    states = [best_q]
    for i in range(n - 1, 0, -1):
        best_q = backpointer[i - 1][best_q]  # backpointer[i-1] corresponds to gap i
        states.append(best_q)
    states.reverse()

    return states


def extract_bursts(
    gaps: list[float],
    states: list[int],
) -> list[BurstSegment]:
    """Extract contiguous burst segments from a state sequence.

    A burst is a maximal contiguous run of gaps with state ≥ 1.
    """
    bursts: list[BurstSegment] = []
    n = len(states)
    i = 0
    while i < n:
        if states[i] >= 1:
            start = i
            max_s = states[i]
            while i < n and states[i] >= 1:
                max_s = max(max_s, states[i])
                i += 1
            end = i - 1
            # The burst spans from event at index 'start' to event at index 'end+1'
            # (since gap k connects event k to event k+1)
            duration = sum(gaps[start:end + 1])
            bursts.append(BurstSegment(
                start_idx=start,
                end_idx=end,
                max_state=max_s,
                total_time_s=duration,
                energy=max_s * duration,
            ))
        else:
            i += 1
    return bursts


def detect_bursts(
    gaps: Sequence[float],
    s: float = 2.0,
    gamma: float = 1.0,
    base_rate: float | None = None,
    max_states: int = 10,
) -> BurstResult | None:
    """Run full burst detection on a sequence of inter-arrival gaps.

    Returns None if there are fewer than 2 gaps (need at least 2 gaps for
    meaningful burst detection).
    """
    n = len(gaps)
    if n < 2:
        return None

    gaps_list = list(gaps)

    if base_rate is None:
        total_time = sum(gaps_list)
        base_rate = n / total_time if total_time > 0 else 1.0

    states = kleinberg_burst(gaps_list, s=s, gamma=gamma, base_rate=base_rate, max_states=max_states)
    bursts = extract_bursts(gaps_list, states)

    max_burst_weight = max(states) if states else 0
    total_burst_energy = sum(b.energy for b in bursts)
    weighted_burst_count = sum(b.max_state for b in bursts)

    return BurstResult(
        gaps=gaps_list,
        states=states,
        bursts=bursts,
        base_rate=base_rate,
        max_burst_weight=max_burst_weight,
        total_burst_energy=total_burst_energy,
        weighted_burst_count=weighted_burst_count,
    )
