#!/usr/bin/env python3
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

import subprocess
import argparse
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
import corner
import emcee


positive = (0.0, np.inf)
plusone = (0.0,1.0)
plusminusone = (-1.0, 1.0)

bounds = {
    'mu' : plusone,
    'mu-library' : plusone,
    'mu-somatic' : plusone,
    'mu-alleles' : positive,
    'theta' : positive,
    'lib-bias' : positive,
    'lib-error-alleles' : positive,
    'lib-error' : plusone,
    'lib-overdisp-hom' : plusone,
    'lib-overdisp-het' : plusone,
    'ref-bias-het' : plusminusone,
    'ref-bias-hom' : plusminusone,
    'ref-bias-hap' : plusminusone
}

def loglike(modelparams, inputparams):
    params = ["--" + str(k) + "=" + str(v) for k,v in modelparams.items()]
    cline = ['dng','loglike'] + params + inputparams
    out = subprocess.run(cline, stdout=subprocess.PIPE)
    if(out.returncode != 0):
        raise ValueError("dng loglike returned an error." +
            "\nThe full command was: " +
            " ".join(cline) +
            "\nCheck stderr for specifics about the error." +
            "\nPerhaps you forgot to provide input?")
    return float(out.stdout.split()[1])

def logprior(params):
    if all([between(v,bounds[p][0],bounds[p][1]) for p,v in params.items()]):
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
    parser = argparse.ArgumentParser(
        description = 'MCMC model parameter estimation for DeNovoGear.',
        epilog = '''All other parameters, such as inputs, will be directly passed to dng loglike.
        Use "dng help loglike" for more information.
        To specify a parameter as a constant that will not be optimized but always passed directly to DeNovoGear,
        specify it with ++ instead of --. For example, `dng-mcmc.py --theta=0.1 ++mu=1e-9`''',
        allow_abbrev = False
        )
    modelparams = parser.add_argument_group("Model Parameters",
        description = '''Model parameters to estimate via MCMC.
        A value must be given for each parameter;
        this value will be used to initialize a maximum likelihood parameter search.
        The parameters that result from this search are then used to initialize the Markov chains.
        To specify a parameter as a constant that is not to be optimized, specify it with ++ instead of --.
        As a consequence, do not use filenames or parameter values that begin with ++.
        It will probably break something.
        ''')
    for param in bounds:
        modelparams.add_argument("--" + param)
    mcmcparams = parser.add_argument_group("MCMC Parameters",
        description = '''Parameters that alter how the MCMC is carried out or the output.
        ''')
    mcmcparams.add_argument("--cornerplot",
        help = "If specified, will save a corner plot to file CORNERPLOT")
    mcmcparams.add_argument("--chainplot",
        help = "If specified, will save a plot of the Markov chains to file CHAINPLOT")
    mcmcparams.add_argument("--mlonly",
        help = "Only perform a maximum likelihood search; do not do an MCMC",
        action = "store_true")
    mcmcparams.add_argument("-t","--threads",
        help = "Number of threads to use [default: 1]",
        default = 1)
    modelargs, otherargs = parser.parse_known_args()
    modelparams = vars(modelargs) #convert Namespace to dict
    progargs = {k : modelparams.pop(k) for k,v in modelparams.items() if k in ["cornerplot","chainplot","mlonly","threads"]}
    modelparams = {str(k).replace('_','-') : float(v) for k,v in modelparams.items() if v is not None}
    otherargs = [str(v).replace('++','--',1) if v[0:2] == '++' else v for v in otherargs]
    return modelparams, progargs, otherargs

def noneify(value):
    if not np.isfinite(value):
        return None
    else:
        return value

def ml_estimate(modelparams, inputparams):
    params, init = zip(*modelparams.items())
    nll = lambda theta, inputparams: -loglike(dict(zip(params, theta)), inputparams) #make a new dictionary with updated values from the optimizer
    bds = [tuple(map(noneify,bounds[p])) for p in params]
    result = op.minimize(nll, init, args=(inputparams), bounds = bds)
    return dict(zip(params, result["x"]))

def lnprob(theta, params, inputparams):
    return logp(dict(zip(params, theta)),inputparams)

'''
Modelparams should be initialized to the initial value of the mcmc.
Nwalkers is the number of walkers and is passed to the ensemble.
We initialize the ensemble in a gaussian ball around the initial values.
Ballradius is the radius of that ball.
'''
def run_mcmc(modelparams, inputparams, threads = 1, nwalkers = 10, ballradius = 1e-3, burnin=10, numsteps = 100):
    params, initvalues = zip(*modelparams.items())
    ndim = len(params)
    init = [(np.array(initvalues) + ballradius * np.random.randn(ndim)) for walker in range(nwalkers)]
    # lnprob = lambda theta, *inputparams: logp(dict(zip(params, theta)), list(inputparams))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = [params,inputparams], threads = threads)
    pos, prob, state = sampler.run_mcmc(init, burnin)
    sampler.reset()
    sampler.run_mcmc(pos,numsteps)
    return sampler

def plot_walkers(sampler, labels, filename):
    fig, axes = plt.subplots(len(labels), figsize=(10,7), sharex=True)
    samples = sampler.chain
    for i in range(len(labels)):
        ax=(axes[i] if len(labels) != 1 else axes) #handle when there is only 1 axis
        ax.plot(samples[:,:,i],"k", alpha = 0.3)
        ax.set_xlim(0,len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1,0.5)
    if len(labels) != 1: #handle when there is only 1 axis
        axes[-1].set_xlabel("step number")
    else:
        axes.set_xlabel("step number")
    plt.savefig(filename)

def plot_corner(sampler, labels, filename):
    corner.corner(sampler.flatchain, labels = labels)
    plt.savefig(filename)

def main():
    modelargs, progargs, otherargs = parseargs()
    ml = ml_estimate(modelparams, otherargs)
    if progargs["mlonly"] is True:
        print(ml)
        exit()
    sampler = run_mcmc(modelparams, otherargs, threads = progargs["threads"])
    labels = list(modelparams.keys())
    if "cornerplot" in progargs:
        plot_corner(sampler, labels, progargs["cornerplot"])
    if "chainplot" in progargs:
        plot_walkers(sampler, labels, progargs["chainplot"])

if __name__ == '__main__':
    main()

