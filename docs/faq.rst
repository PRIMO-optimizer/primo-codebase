Frequently Asked Questions
==========================

How to ...
-----------

Run Examples?

PRIMO includes Jupyter Notebook demos as illustrative :doc:`examples <example>`.  


Get more help?

You can send questions to primo@netl.doe.gov


Troubleshooting
---------------

Missing win32api DLL
    Python 3.8 and perhaps others, may generate an error when running Jupyter on Windows 10 regarding
    missing the win32api DLL. There is a relatively easy fix::

        pip uninstall pywin32
        pip install pywin32==225

Permission error when Running New Versions of Pytest with Stagefright and VS Code
    Some newer versions of Pytest may generate the following error and warnings::

        PermissionError: [WinError 5] Access is denied: 'C:\\Documents and Settings'

        PytestCacheWarning: could not create cache path \\.\NUL\.pytest_cache\v\cache\nodeids: [WinError 1] Incorrect function: 

    As of July, 2024, these warnings and error can be temporally fixed by downgrading Pytest to older versions::
        
        pip uninstall pytest
        pip install pytest== (8.0.0 or older)

    Helpful links: 

        https://github.com/pytest-dev/pytest/issues/12036

        https://github.com/pytest-dev/pytest/issues/11904

Timeout error when using SCIP to solve optimization model
    If  Timeout error is encountered with SCIP solver::
        
        TimeoutExpired: Command [".../scip.exe", "--version"] timed out after 1 second

    As of July, 2024, the error can be temporally fixed by executing the following command in the terminal::

        scip --version


    Helpful link: 
        https://github.com/Pyomo/pyomo/issues/3064

