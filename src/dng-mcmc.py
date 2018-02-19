#!usr/bin/env python3
#
# Copyright (c) 2018 Reed A. Cartwright
# Copyright (c) 2018      Adam J. Orr
#
# Authors:  Reed A. Cartwright <reed@cartwrig.ht>
#           Adam J. Orr <adamjorr@gmail.com>
#
# This file is part of DeNovoGear.
#
# DeNovoGear is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np
import emcee
import subprocess

positive = (0.0, np.inf)
plusone = (0.0,1.0)
plusminusone = (-1.0, 1.0)

bounds = {
    'mu' : plusone,
    'mu-library' : plusone,
    'mu-somatic' : plusone,
    'mu-entropy' : positive,
    'theta' : positive,
    'lib-bias' : positive,
    'lib-error-entropy' : positive,
    'lib-error' : plusone,
    'lib-overdisp-hom' : plusone,
    'lib-overdisp-het' : plusone
    'asc-het' : plusminusone,
    'asc-hom' : plusminusone,
    'asc-hap' : plusminusone
}

def loglike(modelparams, inputparams):
    out = subprocess.run(['dng','loglike'] + params + inputparams, stdout=subprocess.PIPE)
    return float(out.stdout.split()[1])

def logprior(params):
    if all([between(v,bounds[p][0],bounds[p][1]) for p,v in params]):
        return 0.0
    else:
        return -np.inf

def logp(modelparams, inputparams):
    lp = logprior(modelparams)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + loglike(modelparams, inputparams)

def between(param, lowerlim, upperlim):
    return lowerlim <= param <= upperlim

def parseargs():
    #TODO

def main():
    print

if __name__ == '__main__':
    main()

