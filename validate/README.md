# Comparing against GITM's Fortran FTA model

GITM runs the AU/AL model through `src/ModFtaModel.f90` in
[GITMCode/Electrodynamics](https://github.com/GITMCode/Electrodynamics).
`fta_model_aual.py` is kept in step with it. To check that the two still agree, run:

    validate/run.sh <path to an Electrodynamics checkout>

This builds the Fortran model with gfortran and runs both versions at the
AU, AL pairs in `points.txt`. It then compares LBHL, LBHS, energy flux, average
energy and the polar cap mask.

The Fortran and Python versions read their own coefficient files, so differences
should be caught. If needed, set `PYTHON` to choose the interpreter,
e.g. `PYTHON=python3 validate/run.sh ../Electrodynamics`

`points.txt` covers each branch of the AU/AL limiter, AU clamped at both ends,
AL on either side of 500 nT (the split between the two fits), and three points
where the band width and offset adjustments are used.

