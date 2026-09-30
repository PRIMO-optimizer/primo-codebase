#################################################################################
# PRIMO - The P&A Project Optimizer was produced by the National Energy
# Technology Laboratory (NETL).
#
# NOTICE. This Software was developed under funding from the U.S. Government
# and the U.S. Government consequently retains certain rights. As such, the
# U.S. Government has been granted for itself and others acting on its behalf
# a paid-up, nonexclusive, irrevocable, worldwide license in the Software to
# reproduce, distribute copies to the public, prepare derivative works, and
# perform publicly and display publicly, and to permit others to do so.
#################################################################################

"""
This module contains a class that allows experimenting with different
Gurobi features.
"""

# Standard libs
import logging

# Installed libs
import pandas as pd
import pyomo.environ as pyo
from pyomo.contrib.solver.solvers.gurobi.gurobi_persistent import GurobiPersistent

# User-defined libs
from primo.data_parser.default_data import WELL_BASED_METRICS, WELL_PAIR_METRICS
from primo.opt_model.model_with_clustering import PluggingCampaignModel

LOGGER = logging.getLogger(__name__)


class GurobiSolver:
    """Solves the optimization model with Gurobi"""

    def __init__(self, model: PluggingCampaignModel, writer_type: str = "new"):
        """Creates an instance of the solver object

        Parameters
        ----------
        model : PluggingCampaignModel
            Optimization model
        writer_type : str, optional
            Type of Gurobi solver writer. Allowed values are
            "old" and "new", by default "new"
        """
        self.model = model
        self.formulation_type = model.model_inputs.config.efficiency_formulation
        if writer_type == "old":
            self.solver = pyo.SolverFactory("gurobi_persistent")
        else:
            # This is the new Gurobi solver writer. Supposed to be faster
            self.solver = GurobiPersistent()
            self.solver.config.raise_exception_on_nonoptimal_result = False

        self.solver.set_instance(self.model)
        self.pm_to_gb = self.solver._pyomo_var_to_solver_var_map
        self.solve_result = None

    @property
    def gurobi_model(self):
        """Returns the Gurobi model"""
        # pylint: disable = protected-access
        return self.solver._solver_model

    def set_branch_priorities(
        self,
        cluster_vars_priority: int = 10000,
        num_wells_vars_priority: int = 1000,
        zone_vars_priority: int = 100,
    ):
        """
        Sets branching priorities on decision variables. Higher the priority
        value, higher the chances of selecting the variable for branching.

        Parameters
        ----------
        cluster_vars_priority : int, optional
            Priority on `select_cluster` binary variables, by default 10000

        num_wells_vars_priority : int, optional
            Priority on `num_wells_var` binary variables, by default 1000

        zone_vars_priority : int, optional
            Priority on `select_zone` binary variables, by default 100
        """
        m = self.model

        for c in m.set_clusters:
            self.solver.set_var_attr(
                m.cluster[c].select_cluster, "BranchPriority", cluster_vars_priority
            )
            for v in m.cluster[c].num_wells_var.values():
                self.solver.set_var_attr(v, "BranchPriority", num_wells_vars_priority)

            for metric in m.cluster[c].component_data_objects(pyo.Block):
                if not hasattr(metric, "select_zone"):
                    # Not a zone formulation block, so skip
                    continue

                for v in metric.select_zone.values():
                    self.solver.set_var_attr(v, "BranchPriority", zone_vars_priority)

        self.solver.update()  # Updates the solver var attributes in solver model

    def set_partition_number(self):
        """
        Sets partition number for all variables in order to use the
        partition heuristic
        """
        # For variables that are not defined inside the cluster block,
        # set the partition value as 0
        for v in self.model.component_data_objects(pyo.Var, descend_into=False):
            try:
                self.solver.set_var_attr(v, "Partition", 0)

            except KeyError:
                # NOTE: the old writer defines a gurobi var for all pyomo vars.
                # However, the new writer defines a gurobi var only if it is used
                # in the optimization model. That's why, the variable would not be
                # available in the ComponentMap, so we see a KeyError.
                pass

        # For variables that are defined inside the cluster block,
        # set the partition number to be non-zero
        counter = 1
        for c in self.model.set_clusters:
            for v in self.model.cluster[c].component_data_objects(pyo.Var):

                try:
                    self.solver.set_var_attr(v, "Partition", counter)

                except KeyError:
                    # Same reason as the above
                    pass

            counter += 1

        self.solver.update()

    def make_efficiency_constraints_lazy(self, lazy_level: int = 1):
        """Makes a few constraints related to efficiency calculations lazy

        Parameters
        ----------
        lazy_level : int, optional
            Degree of "laziness". See Gurobi documentation., by default 1
        """
        lazy_metrics = WELL_BASED_METRICS + WELL_PAIR_METRICS
        efficiency_blks = []

        try:
            for blk in self.model.cluster.values():
                eff_model = blk.efficiency_model
                efficiency_blks += list(eff_model.component_data_objects(pyo.Block))

        except AttributeError:
            LOGGER.warning(
                "Efficiency calculation constraints are not present in "
                "the model. No constraint is made lazy."
            )
            return

        for eff_blk in efficiency_blks:
            if eff_blk.name.split(".")[-1] not in lazy_metrics:
                # Do not make constraints in this block lazy
                continue

            if self.formulation_type == "Max Scaling":
                for constr in eff_blk.calculate_score.values():
                    self.solver.set_linear_constraint_attr(constr, "Lazy", lazy_level)

        self.solver.update()

    def change_variable_domain(self, to_domain: str, var_list: list[str] | None = None):
        """Changes the domain of certain binary variables

        Parameters
        ----------
        to_domain : str
            The new domain name. Accepted values are "binary", "unit_interval".

        var_list : list[str] | None, optional
            List of variables. Allowed values include `select_cluster`,
            `select_well`, `num_wells_var`, and `select_zone`.
            If `None`, then the domain of all the above variables
            will be changed, by default None
        """
        if to_domain == "binary":
            domain_func = pyo.Binary
        elif to_domain == "unit_interval":
            domain_func = pyo.UnitInterval
        else:
            raise ValueError(f"Unrecognized domain type {to_domain}")

        supported_var_list = [
            "select_cluster",
            "select_well",
            "num_wells_var",
            "select_zone",
        ]

        if var_list is None:
            LOGGER.warning(
                f"Variables names are not specified. Changing the "
                f"domain of these variables {supported_var_list}"
            )
            var_list = supported_var_list

        for v in var_list:
            LOGGER.info(
                f"Changing the domain of {v} variables to {domain_func}, if it exists"
            )
            if v in ("select_cluster", "select_well", "num_wells_var"):
                for blk in self.model.cluster.values():
                    getattr(blk, v).domain = domain_func

            elif v == "select_zone":
                for blk in self.model.cluster.values():
                    for sub_blk in blk.component_data_objects(pyo.Block):
                        if hasattr(sub_blk, "select_zone"):
                            getattr(sub_blk, v).domain = domain_func

            else:
                raise ValueError(
                    f"Unrecognized variable type {v}. Supported variables are {supported_var_list}"
                )

    def set_pool_ignore(self):
        """
        Sets the PoolIgnore attribute on select_well binary variables to 1.
        This way, two campaigns that choose same set of clusters and the same number
        of wells in each cluster, but differ in the choice of wells are not
        distinguished. Campaigns are distinguished only if they choose
        different set of clusters and/or different number of wells in each cluster.
        """
        for v in self.model.well_selected.values():
            self.solver.set_var_attr(v, "PoolIgnore", 1)

        for blk in self.model.cluster.values():
            for v in blk.select_well.values():
                self.solver.set_var_attr(v, "PoolIgnore", 1)

        self.solver.update()

    def add_binary_cut(self):
        """Adds a constraint to eliminate the current optimal solution"""
        m = self.model
        if not hasattr(m, "campaign_elimination_cuts"):
            m.campaign_elimination_cuts = pyo.ConstraintList()

        data = {"pyomo_var": [], "var_value": []}
        for c in m.set_clusters:
            try:
                data["pyomo_var"] += list(m.cluster[c].num_wells_var.values())
                data["var_value"] += [
                    round(v.value) for v in m.cluster[c].num_wells_var.values()
                ]
            except TypeError:
                LOGGER.warning(
                    "Model is not initialized. Returning without adding a cut."
                )
                return

        data = pd.DataFrame(data)
        vars_with_0_value = data[data["var_value"] == 0]
        vars_with_1_value = data[data["var_value"] == 1]

        if len(vars_with_0_value) + len(vars_with_1_value) != len(data):
            raise RuntimeError("Some binary variables have non 0/1 values")

        num_cuts = len(m.campaign_elimination_cuts)
        LOGGER.info(f"Adding a cut to eliminate solution {num_cuts + 1}")

        m.campaign_elimination_cuts.add(
            len(vars_with_0_value)
            - vars_with_0_value["pyomo_var"].sum()
            + vars_with_1_value["pyomo_var"].sum()
            <= len(data) - 1
        )

        try:
            # Works with the new writer, does not work with the old writer
            self.solver.add_constraints([m.campaign_elimination_cuts[num_cuts + 1]])
        except AttributeError:
            # This works with the old writer
            self.solver.add_constraint(m.campaign_elimination_cuts[num_cuts + 1])

        self.solver.update()

    def reset_solver_params(self):
        """Resets all solver parameters to their default values"""
        self.gurobi_model.resetParams()

    def solve(self, **kwargs):
        """
        Solves the optimization model. Pass Gurobi solver parameters
        as keyword arguments to change the default values.
        """

        for k, v in kwargs.items():
            if k != "tee":
                # tee is the only non-Gurobi option we allow, so skipping it.
                self.solver.set_gurobi_param(k, v)

        self.solve_result = self.solver.solve(self.model, tee=kwargs.get("tee", False))

    def get_solution_pool_sequentially(self, sol_count: int, **kwargs):
        """
        Enumerates the solution pool sequentially by adding a binary cut

        Parameters
        ----------
        sol_count : int
            Size of the solution pool

        Returns
        -------
        pandas.DataFrame
            Solutions as arranged as columns in the DataFrame
        """
        data = {v.name: [] for v in self.pm_to_gb}
        data["objective_function"] = []
        objective_func = next(self.model.component_data_objects(pyo.Objective))
        sc = 0

        while sc < sol_count:
            self.solve(**kwargs)
            for v in self.pm_to_gb:
                data[v.name].append(v.value)

            data["objective_function"].append(objective_func.expr())
            self.add_binary_cut()
            sc += 1

        return pd.DataFrame(data).transpose()

    def get_solution_pool(self):
        """
        Retrieves the solution pool from the solver model, and returns it
        as a DataFrame


        Returns
        -------
        pandas.DataFrame
            Solutions as arranged as columns in the DataFrame
        """
        gb_solver = self.gurobi_model
        sel_well_values = {v.name: [] for v in self.pm_to_gb}
        sel_well_values["objective_function"] = []
        gurobi_version = self.solver.version()[0]

        obj_value_name = "PoolObjVal" if gurobi_version <= 12 else "PoolNObjVal"
        var_value_name = "Xn" if gurobi_version <= 12 else "PoolNX"

        for n in range(gb_solver.SolCount):
            gb_solver.Params.SolutionNumber = n
            sel_well_values["objective_function"].append(
                gb_solver.getAttr(obj_value_name)
            )

            for v in self.pm_to_gb:
                sel_well_values[v.name].append(self.pm_to_gb[v].getAttr(var_value_name))

        return pd.DataFrame(sel_well_values).transpose()

    def solve_with_lazy_constraints(self, lazy_level: int = 1, **kwargs):
        """Solves the optimization by making a few constraints lazy

        Parameters
        ----------
        lazy_level : int, default=1
            Accepted values are 1, 2, and 3. Higher value adds
            violated lazy constraints to the model aggressively.
        """
        LOGGER.info("Attempting to find a solution with the full model")
        self.solve(NodeLimit=1, **kwargs)

        LOGGER.info("Setting a few constraints in the efficiency model as lazy")
        self.make_efficiency_constraints_lazy(lazy_level=lazy_level)

        self.reset_solver_params()
        self.solve(**kwargs)
