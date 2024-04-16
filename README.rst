
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. image:: https://readthedocs.org/projects/hypatia-py/badge/?version=latest
    :target: https://hypatia-py.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
    
.. image:: https://img.shields.io/pypi/v/hypatia-py
    :target: https://pypi.org/project/hypatia-py/
   
.. image:: https://badges.gitter.im/Hypatia-py/community.svg
    :target: https://gitter.im/Hypatia-py/community?utm_source=share-link&utm_medium=link&utm_campaign=share-link
    :alt: Documentation Status
    
.. image:: https://zenodo.org/badge/434963275.svg
   :target: https://zenodo.org/badge/latestdoi/434963275

.. image:: https://raw.githubusercontent.com/SESAM-Polimi/MARIO/767d2c0e9e42ae0b6acf7c3a1cc379d7bcd367fa/doc/source/_static/images/polimi.svg
   :width: 200
   :align: right

********
Hypatia
********
An Operation and Planning Energy System Modelling Framework


What is it
-----------
Hypatia is an open source modelling framework written in Python that provides
a technology-rich basis for optimizing both the operation and planning mode of
the energy systems in short-term and long-term time horizons. Hypatia is able
to analyze various energy transition scenarios based on different policies such
as coal phase out, carbon taxes, renewable incentives and other national and
international pledges for the possible future energy systems.

Quickstart
----------
There are different ways to install hypatia software on your machine. The fastest one is through pip:

In case that you are using pip, it is suggested to create a new environment to avoid conflicts of the other packages.
To create a new environment, you should use *Anaconda Prompt*:

.. code-block:: bash

    conda create -n hypatia python=3.8

If you create a new environment for hypatia, you need to activate the environment each time you want to use it, by writing
the following line in *Anaconda Prompt*:

.. code-block:: bash

    conda activate hypatia

After activating the environment, you need to install **CVXPY, version 1.1.18**:

.. code-block:: bash

    conda install -c conda-forge cvxpy=1.1.18 

Then, you can download the code version you are interesting in and save in any folder of your local disk, after unzipping it. After that, open the *Anaconda Prompt*, activate the enviroment of hypatia and digit: 

.. code-block:: bash

    cd "path"
    
"path" of local folder in which you placed the Hypatia folder you just downloaded e.g. C:\Users\YourUser\GitHub\Hypatia. 
Inside the path digit:

.. code-block:: bash

    python setup.py bdist_wheel

FInally, the last command:

.. code-block:: bash

    pip install -e .


Most of the open source solvers that are supported by CVXPY (the optimization library used in Hypatia), will be installed
automatically with the software. For the commercial solvers, you should follow the specific installation methods. 


Python module requirements
--------------------------
Some of the key packages that Hypatia relies on are:

#. `Pandas <https://pandas.pydata.org/>`_
#. `Numpy <https://numpy.org/>`_
#. `Plotly <https://plotly.com/>`_
#. `Cvxpy <https://pypi.org/project/cvxpy/>`_ (domain-specific language)

Hypatia supports different **Open Source** and **Commercial** solvers like:

* `CBC <https://projects.coin-or.org/Cbc>`_
* `GLPK <https://www.gnu.org/software/glpk/>`_
* `OSQP <https://osqp.org/>`_
* `ECOS <https://www.embotech.com/ECOS>`_
* `CVXOPT <http://cvxopt.org/>`_
* `SCS <https://github.com/cvxgrp/scs>`_
* `CPLEX <https://www.ibm.com/products/category/business/commerce>`_
* `GUROBI <https://www.gurobi.com/>`_


.. note::
   * This project is under active development.


License
-------

.. image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
    :target: https://www.apache.org/licenses/


This work is licensed under `Apache 2.0 <https://www.apache.org/licenses/>`_

