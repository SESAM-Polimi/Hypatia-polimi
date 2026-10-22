.. _formulation:

#################
Model formulation
#################

This part describes the optimisation problem Hypatia builds: its sets,
variables and parameters, the objective function and every constraint.
Each equation is followed by the class of the source code that implements it
(:ref:`implementation-map` lists them all), so the text can be checked
against the code and the code read with the text at hand.

The formulation was written from the code (October 2026) and checked against
it equation by equation. Where the code does something that looks
questionable, the text says so in a *Review note*; the notes are collected in
:ref:`review-notes`. They describe what the model does today, not what it
should do.

Hypatia is a linear program (LP) unless modular capacity or one of the
optional constraints with on/off decisions is used, in which case it becomes
a mixed-integer linear program (MILP). It is written with CVXPY, which
passes the problem to the chosen solver.

.. toctree::
   :maxdepth: 2

   nomenclature
   objective
   balances
   capacity
   production
   storage
   emissions
   lines
   optional
   implementation
   review_notes
