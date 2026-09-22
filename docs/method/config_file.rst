Config File
===========

What is the Config File?
------------------------
PRIMO utilizes a .json config file to define priority metrics, efficiency metrics, and constraints. Additionally, it allows users to specify the default paths 
for input data files and output files.

Through the config file, user-provided information about priority metrics, efficiency metrics, and constraints is used to generate corresponding checkboxes and sliders, 
as explained in Section :doc:`How to Provide Inputs to PRIMO <primo_input>`. 

- **Name**: Name of the factor or constraint displayed after the checkbox
- **Default value**: Initial value assigned to the factor; determines the starting position of the slider
- **Value range (provide both the upper and lower thresholds; typically 0---100 for priority and efficiency metrics)**: Defines the range within which the factor's value can be adjusted; the slider's range is determined by the upper and lower thresholds
- **Increment of the value**: Minimum value change possible with one movement of the slider; if the default value falls below this increment, it is rounded up accordingly

Once all necessary information for metrics and constraints is provided in the config file, users can further adjust existing metrics and constraints using checkboxes 
and sliders as shown in the :doc:`Metrics <primo_input>` and :doc:`Constraints <primo_input>` sections. 
Alternatively, users can update the config file directly to make changes.

Information associated with priority metrics, efficiency metrics, and constraints is organized into separate sections within the config file
 as shown in Table below.

.. list-table:: 
        :widths: 25 75
        :header-rows: 1

        * - Input
          - Field Name
        * - Priority metric
          - impact_weights
        * - Efficiency metric
          - efficiency_weights
        * - Constraint
          - program_constraints  

By specifying the path of input data files in the config file, users can designate data files without relying on the file path widget, as presented in the :doc:`Data Files <primo_input>` 
section. The file path widget reads the default file paths documented in the config file, eliminating the need for users to provide file paths again; however, 
the file path widgets are retained to allow users to update file paths if necessary.

Users can specify the path and name of the output file by providing corresponding information in the config file.

How to Construct or Modify the Config File
------------------------------------------
Changes to metrics, constraints, and file paths should be made in their respective fields within the config file. 

Add a Priority Factor or Efficiency Factor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To add a priority factor or efficiency factor, users should write the information into the *config.json* file, following the format below: ::

    "Name of the factor": {
        "default": int, # default weight of the factor, an integer
        "min_val": int, # lower bound of the weight, an integer
        "max_val": int, # upper bound of the weight, an integer
        "incr": int, # increment of the weight, an integer
        "sub_weights" : {# optional if sub-factors are applicable 
            "Name of the sub-factor 1": {
                "default": int, # default weight of the sub-factor, an integer
                "min_val": int, # lower bound of the weight for the sub-factor, an integer
                "max_val": int, # upper bound of the weight for the sub-factor, an integer
                "incr": int, # increment of the weight for the sub-factor, an integer
            },
            "Name of the sub-factor 2": {
                "default": int, # default weight of the sub-factor, an integer
                "min_val": int, # lower bound of the weight for the sub-factor, an integer
                "max_val": int, # upper bound of the weight for the sub-factor, an integer
                "incr": int, # increment of the weight for the sub-factor, an integer
            }
        }
    }

Add a Constraint 
^^^^^^^^^^^^^^^^

.. _addconstraints:

Similarly, when adding a constraint, users should provide the information following the format below: ::

    "Name of the constraint": {
        "default": int, # default value of the constraint, an integer
        "min_val": int, # lower bound of the constraint value, an integer
        "max_val": int, # upper bound of the constraint value, an integer
        "incr": int, # increment of the constraint value, an integer
    }

Update an Existing Metric or Constraint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To modify the default value, value range, or increment of an existing priority factor or constraint, users should update the respective number 
while maintaining the format shown in Section :ref:`Add a Constraint. <addconstraints>`

Remove an Existing Priority Factor, Efficiency Factor, or Constraint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If users wish to exclude an existing priority factor, efficiency factor, or constraint, all associated information must be deleted 
from the *config.json* file entirely. This includes the *"Name of the factor or constraint"* and any information within the `{ }` that follows. 

Add or Modify Input File Path and Output File Path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When specifying paths for input and output files, users should follow the format below: ::

    "Name for calling the file": "file\\path\\file_name.file_extension"

Example of a Config File
-------------------------
Below is an example demonstrating the structure and content of a *config.json* file: ::

    {
        "efficiency_weights": {
            "Priority factor 1": {
                "default": 50,
                "min_val": 0,
                "max_val": 100,
                "incr": 5
            },
            "Priority factor 2": {
                "default": 50,
                "min_val": 0,
                "max_val": 100,
                "incr": 5
            }
        },
        "impact_weights": {
            "Well History": {
                "default": 95,
                "min_val": 0,
                "max_val": 100,
                "incr": 5,
                "sub_weights": {
                    "Compliance [Yes/No]": {
                        "default": 50,
                        "min_val": 0,
                        "max_val": 100,
                        "incr": 5
                    },
                    "Incident [Yes/No]": {
                        "default": 50,
                        "min_val": 0,
                        "max_val": 100,
                        "incr": 5
                    }
                }
            },
            "Owner Well-Count": {
                "default": 5,
                "min_val": 0,
                "max_val": 100,
                "incr": 5
            }
        },
        "program_constraints": {
            "5-year Production Constraint": {
                "default": 500,
                "min_val": 0,
                "max_val": 1000,
                "incr": 100
            },
            "Owner Well Count Constraint": {
                "default": 1,
                "min_val": 1,
                "max_val": 20,
                "incr": 1
            }
        },
        "Well Data": "C:\\Users\\Downloads\\Well_data.xlsx"
    }


.. note::
    Config files should be placed in the same folder as the main working file.

.. note::
    The sum of weights for all parent priority and efficiency factors should be 100. Likewise, for any selected parent factor, 
    the total weights of all corresponding sub-factors should also sum to 100.