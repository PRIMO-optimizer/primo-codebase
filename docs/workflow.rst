Workflow
========

PRIMO is an executable optimization-based decision-support tool specifically designed to support states engaged in well plugging campaigns. It operates by receiving input data concerning MCWs, along with well assessment metrics, 
and user-defined project preferences. In return, users obtain outputs consisting of MCW rankings and recommendations for P&A 
projects. The workflow of PRIMO is depicted in :numref:`workflow-figure`

.. _workflow-figure:

.. figure:: _static/WorkflowFigure.PNG
    :width: 800
    :align: center

    PRIMO design and targeted workflow

Based on :ref:`inputs from users <PRIMO Inputs>`, PRIMO proceeds to :ref:`rank MCWs <Ranking MCWs>`, :ref:`cluster MCWs <Clustering>`, :ref:`execute the optimization process <Optimization>`, and :ref:`calculate efficiency scores of P&A projects <Calculating Project Efficiency Score>` to provide the :ref:`results <PRIMO outputs>`.


.. _PRIMO Inputs:

PRIMO Inputs
------------
PRIMO requires the following information from the user:

- MCW information

    * Location
    * Age
    * Depth
    * Production volume
    * etc

- Priority metric

    * Priority factors and sub-priority factors (if applicable) 
    * Weight assigned to each factor 
    * Data necessary for priority score calculation  

- Efficiency metric

    * Efficiency factors and sub-efficiency factors (if applicable)
    * Weight assigned to each factor
- State-wide program constraints




For detailed instructions on how to import these inputs into PRIMO, refer to Section :doc:`How to Provide Inputs to Primo <method/primo_input>`.



.. _Ranking MCWs:

Ranking MCWs
------------
PRIMO calculates the priority score for each well based on the priority metric provided by the user. This score is computed under each factor defined in the 
metric. The total priority score for a well is the sum of its scores across all factors. For a detailed explanation of how these scores are calculated, 
refer to Section :doc:`Priority Score Calculation <method/priority_score_calculation>`.


.. _Clustering:

Clustering
----------
In the clustering step, wells are grouped into distinct clusters based on their characteristics such as location, age, and depth. 
Each cluster represents a potential P&A project candidate, where one or more wells can be selected to form a P&A project. The clustering step intends to 
reduce the computation time of the optimization step.

For more details on the clustering methodology employed, please refer to Section - :doc:`Clustering Methodology <method/clustering>`.


.. _Optimization:

Optimization
------------
An optimization problem is developed and solved based on the priority scores of wells and the user-provided state-wide program constraints to identify the 
optimal P&A projects. The objective function is set to maximize the total priority score of all selected wells, where one project is 
chosen from one cluster obtained in the clustering step. 

For a detailed formulation of the optimization model, please refer to Section :doc:`Optimization Model for Efficient P&A Campaigns <model_library/index>`.


.. _Calculating Project Efficiency Score:

Calculating the Project Efficiency Score
----------------------------------------
In this step, the efficiency score for each selected P&A project, determined in the optimization step, is computed using the efficiency metric 
supplied by the user. Examples of how the efficiency score is calculated can be found in Section :doc:`Project Efficiency Score Calculation <method/efficiency_score_calculation>` .


.. _PRIMO outputs:

PRIMO Outputs
-------------
PRIMO provides the following results to assist users in making decisions:

- MCW ranking results

    * Well-ranking results
    * Breakdown of priority scores for each well across priority factors

- Recommended high-impact and high-efficiency P&A projects

    * Summary of recommended P&A projects

        * Number of wells in each project
        * Project impact score
        * Project efficiency score
        * Estimated project cost

    * Details on wells included in each project
    * Breakdown of efficiency scores for projects across efficiency factors

These results are displayed through tables and maps, and users have the option to export detailed result information as Excel files for further analysis.
Users are able to choose the folder where they would like the output files to be located by providing the path 
in the :doc:`config file <method/config_file>`. 
