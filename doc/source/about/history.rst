.. _history:

History and credits
===================

Hypatia was created in the `SESAM group <http://sesam.polimi.it/>`_ of the
Department of Energy of Politecnico di Milano by Negar Namazifard and
Mohammad Amin Tahavori, under the scientific coordination of Emanuela
Colombo, with the contribution of `EnergyVille <https://www.energyville.be/>`_
(KU Leuven, VITO, imec and UHasselt), where Pieter Vingerhoets and Erik
Delarue co-coordinated the work. The original repository is
`SESAM-Polimi/hypatia <https://github.com/SESAM-Polimi/hypatia>`_.

It was designed as a model that would

* read its inputs from Excel files, so that they can be prepared and checked
  without programming;
* cover both the hourly operation of an energy system and its investments
  over many years, with annuities based on the economic lifetime and
  interest rate of each technology;
* represent many kinds of technologies (supply, conversion, transmission,
  storage) and several sectors together (power, heat, transport, hydrogen);
* model one region or many, with trade of every carrier between regions;
* accept any time resolution within a year;
* keep a transparent, open source code written with `CVXPY
  <https://www.cvxpy.org>`_ [Diamond2016]_, a language close to the
  mathematical formulation, rather than the more common Pyomo.

Hypatia draws on OSeMOSYS [Howells2011]_, Calliope
[Pfenninger-Pickering2018]_ and TIMES [Loulou2005]_.

This repository
---------------

`SESAM-Polimi/Hypatia-polimi <https://github.com/SESAM-Polimi/Hypatia-polimi>`_
continues the development at SESAM, where Hypatia is used for research on
national energy scenarios and for teaching. The code was restructured in
2022 by Amedeo Felice Alberio, Gianluca Pellecchia and Matteo Rocco.
Contributors to this branch include:

* Khaled Gad: code development and maintenance;
* Gianluca Pellecchia: code development;
* Francesco Cruz Torres: code development;
* Tommaso Zacconi: storage (depth of discharge, cyclic boundary) and
  emission constraints;
* Luigi Malco Rossi: modular unit commitment;
* Nicolò Stevanato: code maintenance, tests and documentation;
* Emanuela Colombo: scientific coordination.

The full list of contributions is in the history of the repository and in
the :ref:`changelog`.

Acknowledgements
----------------

The authors of the original framework thank Professor Emanuela Colombo for
her supervision and support, and Steven Diamond for his help in
understanding and using CVXPY.

References
----------

.. [Diamond2016] Diamond, S., & Boyd, S. (2016). CVXPY: A Python-embedded
   modeling language for convex optimization. *Journal of Machine Learning
   Research*, 17(83), 1-5.
.. [Howells2011] Howells, M., Rogner, H., Strachan, N., Heaps, C.,
   Huntington, H., Kypreos, S., et al. (2011). OSeMOSYS: the open source
   energy modeling system. An introduction to its ethos, structure and
   development. *Energy Policy*, 39(10), 5850-5870.
.. [Pfenninger-Pickering2018] Pfenninger, S., & Pickering, B. (2018).
   Calliope: a multi-scale energy systems modelling framework. *Journal of
   Open Source Software*, 3(29), 825.
.. [Loulou2005] Loulou, R., Remme, U., Kanudia, A., Lehtila, A., &
   Goldstein, G. (2005). *Documentation for the TIMES model, part II*.
   Energy Technology Systems Analysis Programme (ETSAP).
