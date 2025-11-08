"""
Obsolete-Aware Upper Confidence Bound for Constrained Bayesian Optimization.

This module implements an improved UCB approach that:
1. Uses direct violation-based query selection (no S⁻ set computation)
2. Detects and queries obsolete (under-explored) constraints
3. Uses optimistic regret estimation over the feasible region
"""
import torch
import sys

GP_ESTIMATION_OBSOLETE_FACTOR = 2.0


class ObsoleteAwareUCB:
    """
    Obsolete-Aware UCB for constrained optimization.

    Simplifies UCBC by removing S⁻ set logic and directly comparing
    target uncertainty with constraint violations. Adds robustness by
    detecting constraints with poor GP estimates.

    Parameters
    ----------
    mixed : bool, default=True
        If True, queries obsolete constraints alongside target/constraint.
        If False, queries only target or single constraint.

    Attributes
    ----------
    lowest_bound : float
        Tracks the lowest uncertainty bound (legacy, not actively used).
    past_query_idxs : list
        History of queried point indices.
    estimator_idx : int or None
        Index of current best estimate based on optimistic regret.
    mixed : bool
        Whether to use mixed mode with obsolete constraint detection.

    Notes
    -----
    Query Strategy:
    - Computes violation = threshold - constraint_lower_bound for each constraint
    - If max_violation > target_uncertainty: Query that constraint
    - Else: Query target
    - In mixed mode: Also queries constraints where std > 2.0 × target_std

    Key Difference from UCBC:
    - No S⁻ set computation (simpler, more direct)
    - Obsolete constraint detection prevents GP estimation errors
    - Optimistic regret-based estimator instead of lowest bound tracking
    """
    def __init__(self, mixed=True):
        self.lowest_bound = float("inf")
        self.past_query_idxs = []
        self.estimator = None
        self.estimator_idx = None
        self.mixed = mixed

    @staticmethod
    def get_Sminus(
        constraint_lower_f_list,
        constraint_upper_f_list,
        lower_margin_list,
        upper_margin_list,
    ):
        """
        Compute S⁻ set (inherited from UCBC, used for compatibility).

        See UCBC.get_Sminus for details.
        """
        Sminus_cond = torch.ones_like(constraint_lower_f_list[0], dtype=torch.bool)

        for i in range(len(constraint_lower_f_list)):
            constraint_lower_f = constraint_lower_f_list[i]
            constraint_upper_f = constraint_upper_f_list[i]
            lower_margin = lower_margin_list[i]
            upper_margin = upper_margin_list[i]

            Sminus_cond_i = torch.logical_and(
                constraint_lower_f >= lower_margin, constraint_upper_f >= upper_margin
            )
            Sminus_cond = torch.logical_and(Sminus_cond, Sminus_cond_i)

        Sminus_idxs = Sminus_cond.nonzero()
        return Sminus_idxs, Sminus_cond

    @staticmethod
    def get_Sunion(constraint_upper_f_list, upper_margin_list):
        """
        Compute optimistic feasible region S⁺.

        See UCBC.get_Sunion for details.
        """
        Sunion_cond = torch.ones_like(constraint_upper_f_list[0], dtype=torch.bool)

        for i in range(len(constraint_upper_f_list)):
            constraint_upper_f = constraint_upper_f_list[i]
            upper_margin = upper_margin_list[i]

            current_constraint_cond = constraint_upper_f >= upper_margin
            Sunion_cond = torch.logical_and(Sunion_cond, current_constraint_cond)

            if len(current_constraint_cond.nonzero()) == 0:
                print(f"get_Sunion: Constraint {i} is violated!")

        Sunion_idxs = Sunion_cond.nonzero()
        return Sunion_idxs, Sunion_cond

    def get_query(self, threshold_list, target, constraint, other_info):
        """
        Determine next query using direct violation comparison.

        Parameters
        ----------
        threshold_list : list of float
            Constraint threshold values.
        target : dict
            Must contain 'mean_f', 'std_f', 'upper_f', 'lower_f'.
        constraint : dict
            Must contain 'mean_f_list', 'std_f_list', 'upper_f_list', 'lower_f_list'.
        other_info : dict
            Additional information (unused).

        Returns
        -------
        query_idx : torch.Tensor
            Index of point to query.
        query_type : list of int
            [0] for target, [i+1] for constraint i.
            In mixed mode, may include multiple obsolete constraints.
        method : str
            Description of query strategy: "v2" or "BO - CONSTRAINT".
        info : dict
            Margin information (always None for this method).

        Notes
        -----
        Obsolete Detection (mixed=True):
        A constraint is obsolete if: constraint_std > 2.0 × target_std
        These are queried alongside the primary query to prevent GP errors.
        """
        # return query_idx, query_type
        # query_type == 0: target
        # query_type > 0: index of constraint (1-based)
        # query_type < 0: both target and constraint
        target_mean_f = target["mean_f"]
        target_std_f = target["std_f"]
        target_upper_f = target["upper_f"]
        target_lower_f = target["lower_f"]

        constraint_mean_f_list = constraint["mean_f_list"]
        constraint_std_f_list = constraint["std_f_list"]
        constraint_upper_f_list = constraint["upper_f_list"]
        constraint_lower_f_list = constraint["lower_f_list"]

        lower_margin_list = None
        upper_margin_list = None

        # Our approach starts from here!
        upper_margin_list = threshold_list

        # compute S- and S+ w.r.t. self.constraint_approx_margins[-1]
        # \tilde{S}^+ union S-
        Sunion_idxs, Sunion_cond = ObsoleteAwareUCB.get_Sunion(
            constraint_upper_f_list, upper_margin_list
        )

        current_bound = float("inf")

        if len(Sunion_idxs) == 0:
            print("Doing BO on the constraint that is violated")

            max_violation = torch.tensor(0.0)
            query_idx = None
            query_type = []

            violations = []
            for i in range(len(constraint_upper_f_list)):
                constraint_upper_f = constraint_upper_f_list[i]
                upper_margin = upper_margin_list[i]

                violations.append(upper_margin - constraint_upper_f)

            violations = torch.stack(violations)
            # NOTE: this max can be replace by a sum (but then violation should be max(0,violation))
            max_violations, max_violations_idxs = torch.max(violations, dim=0)
            query_idx = torch.argmin(max_violations)
            query_type = [max_violations_idxs[query_idx].numpy() + 1]

            if query_idx is None:
                raise Exception(
                    "Strange: S+ is empty while cannot do BO on the constraint!"
                )
            method = "BO - CONSTRAINT"

        else:  # len(Sunion_idxs) > 0
            print(f"Non empty Sunion: len(Sunion_idxs) = {len(Sunion_idxs)}")

            xplus_splus_idx = Sunion_idxs[torch.argmax(target_upper_f[Sunion_idxs])]
            query_idx = xplus_splus_idx

            query_type = []
            method = "v2"

            current_bound = torch.squeeze(
                target_upper_f[xplus_splus_idx] - target_lower_f[xplus_splus_idx]
            )
            sum_bound = current_bound
            obsolete_constraints = []
            for i, threshold in enumerate(threshold_list):
                constraint_upper_f = constraint_upper_f_list[i]
                constraint_lower_f = constraint_lower_f_list[i]

                violation = torch.squeeze(
                    threshold - constraint_lower_f_list[i][query_idx]
                )
                sum_bound = sum_bound + torch.max(torch.tensor(0.0), violation)

                if current_bound < violation:
                    current_bound = violation
                    query_type = [i + 1]
                    # query_type.append(i + 1)

                if (
                    constraint_std_f_list[i][xplus_splus_idx]
                    > GP_ESTIMATION_OBSOLETE_FACTOR * target_std_f[xplus_splus_idx]
                ):
                    obsolete_constraints.append(
                        i + 1
                    )  # to be added to query_type, so it should be i+1 instead i

            if len(query_type) == 0:
                query_type = [0]  # target has the highest uncertainty
                if self.mixed:
                    # Mixed mode: also query obsolete constraints to prevent GP estimation errors
                    # Obsolete = constraint_std > 2.0 × target_std (under-explored)
                    query_type.extend(obsolete_constraints)
            print("Query type:", query_type)

            estimated_regret = torch.max(target_mean_f + target_std_f) - (
                target_mean_f - target_std_f
            )
            for i, threshold in enumerate(threshold_list):
                constraint_upper_f = constraint_upper_f_list[i]
                constraint_lower_f = constraint_lower_f_list[i]

                violation = torch.nn.ReLU()(
                    torch.squeeze(
                        threshold
                        - (constraint_mean_f_list[i] - constraint_std_f_list[i])
                    )
                )
                estimated_regret = estimated_regret + violation

            self.estimator_idx = Sunion_idxs[torch.argmin(estimated_regret[Sunion_idxs])]

        self.past_query_idxs.append(query_idx)

        return (
            query_idx,
            query_type,
            method,
            {
                "lower_margin_list": None,
                "upper_margin_list": None,
            },
        )
