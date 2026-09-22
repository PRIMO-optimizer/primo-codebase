.. _override_re_optimization_model:

Optimization Model for Override Re-optimization
===============================================
PRIMO provides users the ability to override the solution returned by PRIMO with specific manual changes to include unselected wells in projects, 
remove PRIMO recommended wells from projects or to move wells across projects. Some of these user selections might violate user-defined constraints 
such as ones requiring a project to only include wells within a certain distance from each other. The override re-optimization feature re-solves the 
optimization problem to generate a new set of P&A project recommendations that best meet user-defined constraints while incorporating the user’s 
override selections.

This page focuses on the optimization model used for re-optimizing with overrides. For more details on the override feature itself, see :doc:`../override` section.


This optimization model is used exclusively during the override re-optimization process. For the model used in the standard P&A project recommendation workflow, refer to 
the :ref:`Optimization Model for Efficient P&A Campaigns <mathematical_program_formulation>` section.

Symbols
-------
The following sets, parameters, and variables are defined for the re-optimization model.

.. _override_sets_table:
.. list-table:: **Sets**
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`\mathcal{W}`
     - Set of all wells under consideration.
   * - :math:`\mathcal{C}`
     - | Set of all clusters. Each cluster consists of wells that share similar characteristics. 
       | More details on the clustering methodology can be found in 
       | :doc:`../method/clustering`.
   * - :math:`\mathcal{WP_c}, \forall c \in \mathcal{C}`
     - Set of all well pairs that belong to cluster :math:`\mathit{c}`.
   * - :math:`\mathcal{W}_c \subseteq \mathcal{W},\, \forall c \in \mathcal{C}`
     - Set of wells that belong to cluster :math:`\mathit{c}`.
   * - :math:`\mathcal{O}`
     - Set of all well owners.
   * - :math:`\mathcal{W}_o \subseteq \mathcal{W},\, \forall o \in \mathcal{O}`
     - Set of wells that belong to owner :math:`\mathit{o}`.
   * - :math:`\mathcal{CR}`
     - Set of clusters with reassigned wells.
   * - :math:`\mathcal{R}_c`
     - Set of wells reassigned to cluster :math:`\mathit{c}`.
   * - .. math::

          \mathcal{D}_c = \left\{ (w_1, w_2) \in \mathcal{W}_c \times \mathcal{R}_c
          \;\middle|\; \mathrm{dist}(w_1, w_2) > \text{D}_{\text{max}} \right\},

       .. math::

          \forall c \in \mathcal{CR}
     - | Set of well pairs :math:`(w_1, w_2)` in cluster :math:`\mathcal{CR}`, 
       | where :math:`w_1 \in \mathcal{W}_c`, :math:`w_2 \in \mathcal{R}_c`, 
       | and distance between :math:`\mathcal{w_1}` and :math:`\mathcal{w_2}`
       | is greater than the distance threshold :math:`\text{D}_{\text{max}}`.

.. list-table:: **Parameters**
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`\text{B}`
     - Total budget available for the plugging campaign.
   * - :math:`\text{P}_{cw}`
     - Impact score assigned to each well :math:`\mathit{w}` in cluster :math:`\mathit{c}`.
   * - :math:`\text{M}_n`
     - Cost of plugging a project assuming the project has :math:`\mathit{n}` wells.
   * - :math:`\text{NW}_\text{max}`
     - Maximum number of wells to be chosen in a project.
   * - :math:`\text{NW}_\text{min}`
     - Minimum number of wells to be chosen in a project.
   * - :math:`\text{D}_\text{max}`
     - Maximum distance between two wells to be chosen in a project.  
   * - :math:`\text{OW}_{cw}`
     - Well owner for each well :math:`\mathit{w}` in cluster :math:`\mathit{c}`  
   * - :math:`\text{OW}_\text{max}`
     - Maximum number of wells to be chosen under a well owner.
   * - :math:`\text{SL}^\text{OR}`
     - | Scaling factor for violation on number of wells in a project, owner well count, 
       | and the distance between well pairs.
   * - :math:`\text{SL}^\text{VB}`
     - Scaling factor for the exceeded budget variable :math:`\mathit{VB}`.
   * - :math:`\text{SL}^\text{UB}`
     - Scaling factor for the unused budget variable :math:`\mathit{UB}`.


.. list-table:: **Binary Variables**
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`\mathit{SW_{cw}}`
     - 1, if well :math:`\mathit{w}` is chosen in cluster :math:`\mathit{c}`.
   * - :math:`\mathit{SC_c}`
     - 1, if wells in cluster :math:`\mathit{c}` are chosen; 0, otherwise.
   * - :math:`\mathit{NW_cn}`
     - Binary variables to track the number of wells :math:`\mathit{n}` in cluster :math:`\mathit{c}`.
   * - :math:`\mathit{{DW}_{w_1w_2}}`
     - 1, if well :math:`\mathit{w_1}` and :math:`\mathit{w_2}` are selected and exceed distance threshold.
   * - :math:`\mathit{BST}`
     - 1, if the total well plugging cost exceeds the budget; 0, if there is remaining unused budget.


.. list-table:: **Continuous Variables**
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`\mathit{PC_c}`
     - Total cost of plugging all wells in a project in cluster :math:`\mathit{c}`.
   * - :math:`\mathit{TW_c}`
     - Total number of wells chosen in a project in cluster :math:`\mathit{c}`.
   * - :math:`\mathit{VW_c}^\text{min}`
     - | Amount by which the number of wells chosen in a project in cluster :math:`\mathit{c}` 
       | falls below the minimum required per project.
   * - :math:`\mathit{VW_c}^\text{max}`
     - | Amount by which the number of wells chosen in a project in cluster :math:`\mathit{c}` 
       | exceeds the maximum required per project.
   * - :math:`\mathit{UB}`
     - Amount by which the total available budget is not used.
   * - :math:`\mathit{VB}`
     - Amount by which the cost for plugging wells exceeds the total available budget.
   * - :math:`\mathit{VOW_o}`
     - Number of wells exceed the maximum number of wells selected per owner for owner :math:`{\mathit{o}}`.
   * - :math:`\mathit{\eta_c}^\text{impact}`
     - Impact score of the project in cluster :math:`\mathit{c}`.
   * - :math:`\mathit{\eta_c}^\text{efficiency}`
     - Efficiency score of the project in cluster :math:`\mathit{c}`.
   * - :math:`\mathit{\eta_c}^\text{e}`
     - Efficiency score of the project in cluster :math:`\mathit{c}` for the metric e in set :math:`\mathcal{E}`.
   * - :math:`\mathit{VD_c}`
     - Number of well pairs violated the maximum distance threshold :math:`\text{OW}_\text{max}` in cluster :math:`\mathit{c}`.
   * - :math:`\mathit{VD}`
     - Number of well pairs violated the maximum distance threshold :math:`\text{OW}_\text{max}`.
   * - :math:`\mathit{PEN}`
     - | A penalty variable to represent constraints being violated during the re-optimization, 
       | which will be included in the objective function 

.. _impact_only_model:

Impact-only Optimization Model
------------------------------

**Objective Function**
The objective is to maximize the total impact of selected wells while minimizing constraint violations due to override selection 
by applying weighted penalty terms.

.. math::

   \max \sum_{c \in \mathcal{C}} \mathit{\eta_c}^\text{impact} - \mathit{PEN} - \text{SL}^{\text{UB}} \cdot\mathit{UB}

**Explanation**: The model seeks to maximize the total impact score from selected wells while penalizing violations of exceed budget, 
number of wells selected under the same owner, distance among wells, and project size constraints through the penalty term :math:`\mathit{PEN}`.


**Constraints**

.. math::

   \sum_{w \in \mathcal{W}_c} SW_{cw} = TW_c, \quad \forall \, c \in \mathcal{C}

**Explanation**: This constraint computes the number of wells chosen in cluster :math:`\mathit{c}`.

.. math::

   {TW_c} = \sum_{n = 1}^{|\mathcal{W}_c|} n \cdot {NW_{cn}}, \quad \forall \, c \in \mathcal{C}

**Explanation**: This constraint ensures the appropriate binary variable corresponding to the number of wells :math:`\mathit{TW}_{c}` selected for plugging in cluster :math:`\mathit{c}` is activated.

.. math::

   {SC_c} = \sum_{n = 1}^{|\mathcal{W}_c|} {NW_{cn}}, \quad \forall \, c \in \mathcal{C}

**Explanation**: This constraint ensures at most one :math:`\mathit{NW}_{cn}` is selected.

.. math::

   \sum_{n = 1}^{|\mathcal{W}_c|} \text{M}_n \cdot {NW_{cn}} = {PC_c}, \quad \forall \, c \in \mathcal{C}

**Explanation**: This constraint computes the cost of the project.

.. math::

   \text{B} - \sum_{c \in \mathcal{C}} {PC_c} = UB - VB

**Explanation**: This constraint allows the total cost for plugging wells to exceed the budget by a penalty variable :math:`\mathit{VB}`.

.. math::

   UB \leq B \cdot BST


.. math::
   VB \leq B \cdot (1-BST)

**Explanation**: These constraints ensure that :math:`UB` and :math:`VB` cannot both be non-zero simultaneously.

.. math::

   \eta^\text{impact} = \sum_{c \in \mathcal{C}} \sum_{w \in \mathcal{W}} \text{P}_{cw} \cdot {SW_{cw}}

**Explanation**: This constraint computes the total impact score of all projects, which is then maximized in the objective function.

.. math::

   {SC_c} \cdot \text{NW}_{\min} \leq {\mathit{TW}_c} + {\mathit{VW_c}^\text{min}}, \quad \forall \, c \in \mathcal{C}

**Explanation**: This optional constraint includes a penalty variable :math:`\mathit{VW_c}^\text{min}` to relax the minimum number of wells required
in cluster :math:`\mathit{c}`, if the cluster is selected.

.. math::

   {SC_c} \cdot \text{NW}_{\max} \geq {\mathit{TW}_c} - {\mathit{VW_c}^\text{max}}, \quad \forall \, c \in \mathcal{C}

**Explanation**: This optional constraint includes a penalty variable :math:`\mathit{VW_c}^\text{max}` to relax the maximum number of wells required 
in cluster :math:`\mathit{c}`, if the cluster is selected.

.. math::

    \mathit{SW_{cw_1}} + \mathit{SW_{cw_2}} \le {SC_c} + \mathit{{DW}_{w_1w_2}} \quad \forall \, w_1,w_2 \in \mathcal{D_c}

**Explanation**: This constraint includes a penalty variable :math:`\mathit{{DW}_{w_1w_2}}` to allow the selection of distant well pairs.

.. math::

    \mathit{VD_c} = \sum_{(w_1, w_2) \in \mathcal{D_c}} \mathit{{DW}_{w_1w_2}} \quad \forall \, c \in CR

**Explanation**: This constraint computes the number of distant well pairs in cluster :math:`\mathit{CR}` that violate the distance threshold.

.. math::

    \mathit{VD} = \sum_{c \in \mathcal{CR}} VD_c

**Explanation**: This constraint calculates the total number of well pairs violates the maximum distance threshold across all clusters.

.. math::

    \sum_{c \in \mathcal{C}} \sum_{w \in \mathcal{W}} W_o \cdot \mathcal{x_{cw}} \leq \text{OW}_\text{max} + \mathit{VOW_o} \quad \forall \, o \in O

**Explanation**: This optional constraint includes a penalty variable :math:`\mathit{VOW_o}` to allow the number of wells selected for a specific owner to exceed the maximum allowed limit.

.. math::

     \mathit{PEN} = \mathit{SL}^{\text{VB}} \cdot \mathit{VB} + \mathit{SL}^{\text{OR}} \cdot \left[ \mathit{VD} + \sum_{o \in \mathcal{O}} \mathit{VOW}_o + \sum_{c \in \mathcal{C}} \left( \mathit{VW}_c^{\min} + \mathit{VW}_c^{\max} \right) \right]

**Explanation**: This penalty term aggregates all constraint violations and scales them using weight factors to ensure consistency with the magnitude of impact scores. 
The term includes penalties for exceeding the budget (:math:`\mathit{VB}` ), violating well owner count limits (:math:`\mathit{VOW}_o`), selecting too few or too many wells in a project
(:math:`\mathit{VW}_c^{\min}` and :math:`\mathit{VW}_c^{\max}`), and selecting distant well pairs (:math:`\mathit{VD}` ). These penalties are then minimized in the objective function.


Impact & Efficiency Optimization Model
--------------------------------------

**Objective Function**

.. math::

   max \sum_{c \in \mathcal{C}} \left( \zeta^\text{impact} \cdot \mathit{\eta_c}^\text{impact} + \zeta^\text{efficiency} \cdot \mathit{\eta_c}^\text{efficiency} \right) - \mathit{PEN} - \text{SL}^{\text{UB}} \cdot\mathit{UB}

**Explanation**: The objective of the model is to maximize the impact and efficiency, with penalties applied for constraints violations.

**Constraints**

All override violation-related constraints (budget, well owner count, well distance, project size) are identical to those listed in the :ref:`Impact-only Optimization Model <impact_only_model>` above.
Additional efficiency-related constraints can be found in the :ref:`Impact & Efficiency Optimization Model <impact_efficiency_model>` section.