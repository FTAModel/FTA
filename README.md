# FTA
This is the home of the Feature Tracking of the Aurora Model

There are three models that are included here. All of these models
specify the total energy flux and the average energy of the aurora
as a function of magnetic latitude and magnetic local time.

1. Wu et al. [2021a] (in review): An Auroral Electrojet driven model.

2. Wu et al. [2021b] (to be submitted): An Auroral Upper and Auroral
Lower driven model.

3. Fuller-Rowell and Evans [1987]: A Hemispheric Power driven
model. This model was not created by University of Michigan, it was
created by Tim Fuller-Rowell, who works for NOAA.  The python code
was written by us, but the data file that is used was actually created
by NCAR (Art Richmond and Barbara Emery) for use in the AMIE technique.

The AU/AL driven model help can be looked at using:

./fta_model_aual.py -help

The Fuller-Rowell and Evans model can be run using:

./fta_model_aual.py -fre -hp=XXX

Where XXX can technically be anything, but 0-100 is a pretty ok
range. HP get to a max of around 2,500 during the worst possible
conditions (super, super rare).  The 10 patterns in FR&E are scaled
to the input hemispheric power, so that the strongest pattern, which has
a HP of about 77, is simply scaled up to the given HP if it is larger
than 77.
