"""
Upper Confidence Bound for Constrained Bayesian Optimization (UCBC).

This module implements UCBC with S⁻ set-based constraint handling.
Supports both adaptive (decoupled) and non-adaptive (coupled) query strategies.
"""
import torch
import sys

GP_ESTIMATION_GUARD_FACTOR = 10.0

class UCBC:
    """
    Upper Confidence Bound for Constrained Bayesian Optimization.
    
    Uses S⁻ set membership to decide whether to query the objective or constraints.
    The S⁻ set contains points where both lower and upper confidence bounds 
    satisfy all constraints.
    
    Parameters
    ----------
    adaptive : bool, default=True
        If True (decoupled), intelligently queries either target OR constraint.
        If False (coupled), always queries target AND all constraints together.
    
    Attributes
    ----------
    lowest_bound : float
        Tracks the lowest uncertainty bound across iterations.
    estimator_idx : int or None
        Index of the current best estimate in the domain.
    adaptive : bool
        Query strategy mode.
    
    Notes
    -----
    Query Strategy (adaptive=True):
    - If S⁻ is empty or best point not in S⁻: Query constraint with max violation
    - If S⁻ is non-empty and best point in S⁻: Query target
    
    Query Strategy (adaptive=False):
    - Always queries target and all constraints simultaneously
    """
    def __init__(self, adaptive=True):
        self.lowest_bound = float("inf")
        self.estimator = None
        self.estimator_idx = None
        self.adaptive = adaptive

    @staticmethod
    def get_Sminus(
        constraint_lower_f_list,
        constraint_upper_f_list,
        lower_margin_list,
        upper_margin_list,
    ):
        """
        Compute S⁻ set: points where both lower and upper bounds satisfy constraints.
        
        Parameters
        ----------
        constraint_lower_f_list : list of torch.Tensor
            Lower confidence bounds for each constraint.
        constraint_upper_f_list : list of torch.Tensor
            Upper confidence bounds for each constraint.
        lower_margin_list : list of float
            Lower margin thresholds for each constraint.
        upper_margin_list : list of float
            Upper margin thresholds for each constraint.
        
        Returns
        -------
        Sminus_idxs : torch.Tensor
            Indices of points in S⁻.
        Sminus_cond : torch.Tensor
            Boolean mask for S⁻ membership.
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
        Compute S⁺ ∪ S⁻: optimistic feasible region using upper bounds.
        
        Parameters
        ----------
        constraint_upper_f_list : list of torch.Tensor
            Upper confidence bounds for each constraint.
        upper_margin_list : list of float
            Threshold values for each constraint.
        
        Returns
        -------
        Sunion_idxs : torch.Tensor
            Indices of points in optimistic feasible region.
        Sunion_cond : torch.Tensor
            Boolean mask for membership.
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
        Determine next query point and what to evaluate (target/constraint).
        
        Parameters
        ----------
        threshold_list : list of float
            Constraint threshold values.
        target : dict
            Dictionary with 'upper_f' and 'lower_f' keys containing target bounds.
        constraint : dict
            Dictionary with 'upper_f_list' and 'lower_f_list' keys.
        other_info : dict
            Additional information (unused).
        
        Returns
        -------
        query_idx : torch.Tensor
            Index of point to query.
        query_type : list of int
            What to query: [0] for target, [i+1] for constraint i, 
            or [0, 1, 2, ...] for all (coupled mode).
        method : str
            Description of query strategy used.
        info : dict
            Additional information about margins.
        """
        # return query_idx, query_type
        # query_type == 0: target
        # query_type > 0: index of constraint (1-based)
        # query_type < 0: both target and constraint
        target_upper_f = target["upper_f"]
        target_lower_f = target["lower_f"]
        constraint_upper_f_list = constraint["upper_f_list"]
        constraint_lower_f_list = constraint["lower_f_list"]
        lower_margin_list = None
        upper_margin_list = None

        # Our approach starts from here!
        upper_margin_list = threshold_list

        # compute S- and S+ w.r.t. self.constraint_approx_margins[-1]
        # \tilde{S}^+ union S-
        Sunion_idxs, Sunion_cond = UCBC.get_Sunion(
            constraint_upper_f_list, upper_margin_list
        )

        current_bound = float("inf")

        """
        NOTE: we do not simply query the constraint with the largest constraint_upper_f that is < upper_margin

        different constraints violates at different inputs
        hence, it can happen that all constraints are satisfied at some (different) inputs
        but they are not all satisfied at an input

        The querying strategy:
            find set of inputs where at least 1 constraint violated
            find the violation distance of this input: the maximum violation among all violated constraints at this input
        """

        if len(Sunion_idxs) == 0:
            # doing BO on the constraint
            # NOTE: we cannot do BO (BO may stuck in infeasible solution regions), we must do levelset estimation on the constraint
            # but standard levelset estimation may not work
            # we need to do levelset estimation jointly among all constraints
            # how? point with minimum violation among all constraints?
            print("Doing BO on the constraint that is violated")

            max_violation = torch.tensor(0.0)
            query_idx = None
            query_type = []

            violations = []
            print("DEBUGGGGGG")
            for i in range(len(constraint_upper_f_list)):
                constraint_upper_f = constraint_upper_f_list[i]
                upper_margin = upper_margin_list[i]

                violations.append(upper_margin - constraint_upper_f)
                print(constraint_upper_f.shape)

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

            margin = torch.squeeze(
                target_upper_f[xplus_splus_idx] - target_lower_f[xplus_splus_idx]
            )

            print("Margin:", margin)
            if not self.adaptive:
                query_type = [0] + list(range(1,len(threshold_list)+1))

                method = "both target and constraints"

                # update estimator
                current_bound = margin
                for i, threshold in enumerate(threshold_list):
                    current_bound = max(
                        current_bound, threshold - constraint_lower_f_list[i][query_idx]
                    )

                if current_bound < self.lowest_bound:
                    print(f"Update lowest bound to {current_bound}")
                    self.lowest_bound = current_bound
                    self.estimator_idx = query_idx
                else:
                    print(f"NOT Update lowest bound {self.lowest_bound}")

                return (
                    query_idx,
                    query_type,
                    method,
                    {
                        "lower_margin_list": None,
                        "upper_margin_list": None,
                    },
                )

            lower_margin_list = [
                upper_margin - margin
                for i, upper_margin in enumerate(upper_margin_list)
            ]

            Sminus_idxs, Sminus_cond = UCBC.get_Sminus(
                constraint_lower_f_list,
                constraint_upper_f_list,
                lower_margin_list,
                upper_margin_list,
            )

            # check if xplus_splus in S^+ / S^-
            Sunion_diff_Sminus_cond = torch.logical_and(
                Sunion_cond, torch.logical_not(Sminus_cond)
            )

            is_xplus_splus_in_diffset = Sunion_diff_Sminus_cond[xplus_splus_idx]

            if len(Sminus_idxs) == 0 or is_xplus_splus_in_diffset:
                print(
                    f"Doing LSE on the constraint: len(Sminus_idxs) = {len(Sminus_idxs)}, {is_xplus_splus_in_diffset}"
                )

                max_gap_at_xplus_splus_idx = None  # selected constraint to query
                max_gap_at_xplus_splus = -1e9

                for i, constraint_lower_f in enumerate(constraint_lower_f_list):
                    gap = max(
                        0, threshold_list[i] - constraint_lower_f[xplus_splus_idx]
                    )

                    if gap > max_gap_at_xplus_splus:
                        max_gap_at_xplus_splus = gap
                        max_gap_at_xplus_splus_idx = i

                """
                current_bound = (
                    constraint_upper_f_list[max_gap_at_xplus_splus_idx][
                        xplus_splus_idx
                    ]
                    - constraint_lower_f_list[
                        max_gap_at_xplus_splus_idx
                    ][xplus_splus_idx]
                )
                """
                current_bound = max_gap_at_xplus_splus

                query_type = [max_gap_at_xplus_splus_idx + 1]
                method = "LSE - CONSTRAINT"

            else:
                print(
                    f"Doing BO on the objective: len(Sminus_idxs) = {len(Sminus_idxs)}, {is_xplus_splus_in_diffset}"
                )
                query_type = [0]
                current_bound = (
                    target_upper_f[xplus_splus_idx] - target_lower_f[xplus_splus_idx]
                )
                method = "BO - objective"

                # if the current_bound < constraint_uncertainty_at_xplus_splus_idx / GP_ESTIMATION_GUARD_FACTOR
                # query the constraint just to be sure that we do not get an incorrect GP hyperparameters
                for i, constraint_lower_f in enumerate(constraint_lower_f_list):
                    constraint_upper_f = constraint_upper_f_list[i]
                    uncertainty_constraint_i = constraint_upper_f[xplus_splus_idx] - constraint_lower_f[xplus_splus_idx]
                    if uncertainty_constraint_i > GP_ESTIMATION_GUARD_FACTOR * current_bound:
                        print(f"WARNING: constraint {i} is not estimated well at the input query. Add constraint {i} to query.")
                        query_type.append(i+1)


        if current_bound < self.lowest_bound:
            print(f"Update lowest bound to {current_bound}")
            self.lowest_bound = current_bound
            self.estimator_idx = query_idx
        else:
            print(f"NOT Update lowest bound {self.lowest_bound} vs. current bound {current_bound}")

        # update the target or the constraint with new observation
        print("  Query idx:", query_idx)
        return (
            query_idx,
            query_type,
            method,
            {
                "lower_margin_list": lower_margin_list,
                "upper_margin_list": upper_margin_list,
            },
        )
