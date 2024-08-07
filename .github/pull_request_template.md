# Add or update a Gusto case study
Here is a checklist of things that should be done to add a new case study to
the repository:
- [ ] The case study has been prepared from the case studies template in `templates/template_case_study.py`. This ensures that the case study:
  - [ ] begins with documentation of the case
  - [ ] includes a dictionary of default argument values
  - [ ] is run through a function
  - [ ] follows the standard order of sections:
    1. test case parameters
    2. set up of model objects
    3. initial conditions
    4. run
  - [ ] includes a `__main__` routine with arg-parsing of command line arguments
- [ ] The case study has a quick-to-run test form in the relevant `test_*.py` file, so that it will be run as part of CI
- [ ] A plotting script has been added to the relevant `plotting` directory, with a name that matches the case study script
- [ ] Neat figures have been added to the relevant `figures` directory, with names that match the case study script

# Add or update a plotting script
Here is a checklist of things that should be done to add a new plotting script to the repository:
- [ ] The plotting script has been prepared from the template in `templates/template_plot.py`

<!--
Here is a comment that can include verbose instructions that will not
appear in the template
-->
