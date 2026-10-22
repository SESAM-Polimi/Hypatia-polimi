"""
No simultaneous charge and discharge (option boolean_for_storage).

The binary must constrain every time step of every year. Until October 2026
the slice [y*T : (y+1)*T - 1] left out the last time step of each year (the end
of a Python slice is already excluded), so in that step a storage could charge
and discharge at once.
"""
import contextlib
import io

from hypatia import load_example
from hypatia.backend.Build import BuildModel


def test_binary_covers_every_time_step_of_every_year():
    data = load_example("Minigrid", optional_constraints=["boolean_for_storage"]).get_model_data()
    with contextlib.redirect_stdout(io.StringIO()):
        built = BuildModel(data)
    steps = len(data.settings.time_steps)
    rules = built.constraints_by_class["BooleanForStorage"]
    assert rules, "the example should have a storage with the same input and output carrier"
    # two constraints (charge, discharge) per storage and year, each over all T steps
    assert {rule.shape[0] for rule in rules} == {steps}
    assert len(rules) % (2 * len(data.settings.years)) == 0
