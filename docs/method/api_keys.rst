API Keys
========

PRIMO can utilize the `US Census API <https://api.census.gov/data/key_signup.html>`_ to extract auxiliary information
about well locations such as population density. You will need to provide your own API keys to use this API by signing up at the link above.

These API Keys can be utilized by providing them in a .env file. A .env file is a text file at the root folder of the project formatted as follows::

    CENSUS_KEY="My census key"


.. note::
    The .env file has extension .env and no name. If you run into errors, please confirm the file is not accidentally named .env.txt.