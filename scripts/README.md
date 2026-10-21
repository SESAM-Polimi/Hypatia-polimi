# Run scripts

Ready-to-run scripts for Hypatia models, used in our courses.

| Script | Model |
| --- | --- |
| `run_planning.py` | Planning: several years, Hypatia decides what to build and when. Starts from the example `Planning_1Region`. |
| `run_operation.py` | Operation: one year, capacity given, Hypatia decides how to run it hour by hour. Starts from the example `Operation_1Region`. |

## How to use them

1. Install Hypatia (see the README at the top of the repository).
2. Copy the script you need to a folder of your own.
3. Open it (for example in Spyder) and change the values in the **SETTINGS** section at the top: the model folder, what to minimise, the solver, the optional constraints. The rest of the script does not need changes.
4. Run it all (Spyder: F5) or one cell at a time (Ctrl+Enter).

The first time, the script copies the example model next to itself, so it runs as it is. To run your own model, set `MODEL_FOLDER` to the folder that holds your `sets` and `parameters` folders, and `EXAMPLE = None`.

Results go to `MODEL_FOLDER/results`, one CSV file per quantity, and plots to `MODEL_FOLDER/plots` (HTML files, open them in a browser). The plots need a `config.xlsx` file in the model folder, with names, colours and groups of technologies and carriers; if it is missing, the script creates one for you to fill in.

## What each step does

| Cell | Step |
| --- | --- |
| 1 | Finds the model folder, copying the example if needed. |
| 2 | Creates the model from the set files (regions, years, technologies, carriers). |
| 3 | Optional: writes empty parameter files, the starting point of a new model. |
| 4 | Reads the parameter files. |
| 5 | Solves: least cost, least emissions, or a series of solutions in between. |
| 6 | Saves the results as CSV files. |
| 7 | Draws the plots. |

The documentation explains each step in detail: building a model, the input sheets, the optional constraints and the results.

## Other files

`Tema.json` is a colour theme for Power BI, for students who analyse the CSV results there.
