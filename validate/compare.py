#!/usr/bin/env python

# Compare fta_model_aual.py against the Fortran results written by driver.f90.
# Run from the directory holding points.txt and the f_*.bin files.

import contextlib
import io
import os
import sys
import numpy as np

FtaDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, FtaDir)
import fta_model_aual as fta

fta.DataDir = os.path.join(FtaDir, 'inputs_aual') + '/'
# GITM fills AveE outside of the oval with 2 keV
fta.AveEFill = 2.0

names = ['lbhl', 'lbhs', 'eflux', 'avee', 'polarcap']
nMLTs, nMLats = 96, 120

worst = 0.0
for au, al in np.loadtxt('points.txt'):
    raw = np.fromfile('f_{:d}_{:d}.bin'.format(round(au), round(-al)), dtype='<f8')
    au_f, al_f = raw[0:2]
    # Fortran arrays are (nMLTs, nMLats), column-major
    arrays = raw[2:].reshape(len(names), nMLats, nMLTs)
    fortran = {name: arrays[i].T for i, name in enumerate(names)}

    # (quietly: the limiter prints when it changes AU or AL)
    with contextlib.redirect_stdout(io.StringIO()):
        python = fta.calc_fta_aual(au, al)

    line = '{:5.0f} {:6.0f}  limited to {:6.1f} {:7.1f} (fortran {:6.1f} {:7.1f})'.format(
        au, al, python['au'], python['al'], au_f, al_f)
    for name in names:
        diff = np.max(np.abs(python[name] - fortran[name])) / \
            max(np.max(np.abs(fortran[name])), 1e-30)
        worst = max(worst, diff)
        line += '  {} {:.1e}'.format(name, diff)
    print(line)

print('Worst relative difference : {:.1e}'.format(worst))
if worst > 1e-10:
    print('Python and Fortran DISAGREE')
    sys.exit(1)
print('Python and Fortran agree')
