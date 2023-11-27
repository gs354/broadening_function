import emcee
import numpy as np
import matplotlib.pyplot as plt
import math
import george
from george import kernels
from george.modeling import Model
from astropy.io import fits
from matplotlib import rcParams
import matplotlib.ticker as ticker
import corner
from multiprocessing import Pool
from contextlib import closing
import time
import datetime
import os
import glob
import logging
import re
import sys
import shutil
import pylab
import chains

# import matplotlib as mpl
# mpl.use('pdf')


os.environ["OMP_NUM_THREADS"] = "1"
epat = re.compile(r"^([^e]+)e(.+)$")


def call(
    fname="",
    system="B1EB",
    rvstd=30.568,
    nspec=9,
    par_per_BF=13,
    n_indiv_pars=7,
    ln_wn=-13.82,
    ln_var=-12,
    ln_ls=4,
    nCPUs=8,
    burn_and_resample=True,
    nsteps=500,
    nburn=0,
    nwalkers=150,
    preburn1=5000,
    preburn2=5000,
    acorr=500,
    run=True,
    do_savefig=True,
    do_plot=True,
    do_chains=True,
    do_cornerplots=True,
    monitor_acorr=False,
):
    bf = BF_Fit(
        fname,
        system,
        rvstd,
        nspec,
        par_per_BF,
        n_indiv_pars,
        ln_wn,
        ln_var,
        ln_ls,
        nCPUs,
        burn_and_resample,
        nsteps,
        nburn,
        nwalkers,
        preburn1,
        preburn2,
        acorr,
        run,
        do_savefig,
        do_plot,
        do_chains,
        do_cornerplots,
        monitor_acorr,
    )

    bf.mcmc_call()

    return


def call_mcmc(
    initial_pos,
    nwalkers,
    nsteps,
    npar,
    ln_prob,
    nCPUs,
    chainfile,
    burn_and_resample,
    monitor_acorr,
):
    # original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    backend = emcee.backends.HDFBackend(chainfile)

    if monitor_acorr == True:
        if nCPUs > 1:
            with closing(Pool(processes=nCPUs)) as pool:
                sampler, pos, ac, steps = run_mcmc(
                    initial_pos,
                    nwalkers,
                    nsteps,
                    npar,
                    ln_prob,
                    nCPUs,
                    backend,
                    burn_and_resample,
                    monitor_acorr,
                    pool,
                )
                pool.terminate()
        else:
            sampler, pos, ac, steps = run_mcmc(
                initial_pos,
                nwalkers,
                nsteps,
                npar,
                ln_prob,
                nCPUs,
                backend,
                burn_and_resample,
                monitor_acorr,
                pool=None,
            )

        return sampler, pos, ac, steps

    else:
        if nCPUs > 1:
            with closing(Pool(processes=nCPUs)) as pool:
                sampler, pos = run_mcmc(
                    initial_pos,
                    nwalkers,
                    nsteps,
                    npar,
                    ln_prob,
                    nCPUs,
                    backend,
                    burn_and_resample,
                    monitor_acorr,
                    pool,
                )
                pool.terminate()
        else:
            sampler, pos = run_mcmc(
                initial_pos,
                nwalkers,
                nsteps,
                npar,
                ln_prob,
                nCPUs,
                backend,
                burn_and_resample,
                monitor_acorr,
                pool=None,
            )

        return sampler, pos


def run_mcmc(
    initial_pos,
    nwalkers,
    nsteps,
    npar,
    ln_prob,
    nCPUs,
    backend,
    burn_and_resample,
    monitor_acorr,
    pool=None,
):
    # signal.signal(signal.SIGINT, original_sigint_handler)
    sampler = emcee.EnsembleSampler(
        nwalkers,
        npar,
        ln_prob,
        moves=emcee.moves.StretchMove(a=1.65),
        pool=pool,
        backend=backend,
    )
    start = time.time()

    pos = initial_pos + 1e-4 * np.random.randn(nwalkers, npar)
    if burn_and_resample == True:
        print("Running two initial burn-ins. First burn-in...")
        pos, lnp, _ = sampler.run_mcmc(pos, self.preburn1, progress=True)

        print("Re-sampling walkers and running second burn-in...")
        # Re-sample positions of walkers in a ball around position of the best walker in the previous run
        pos = pos[np.argmax(lnp)] + 1e-4 * np.random.randn(nwalkers, npar)
        sampler.reset()
        pos, _, _ = sampler.run_mcmc(pos, self.preburn2, progress=True)
        sampler.reset()

    print("\n\t Running production chain... \n")
    pos, lnp, _ = sampler.run_mcmc(pos, nsteps, progress=True)

    if monitor_acorr == True:
        max_n = 250000
        xtra_steps = 50000
        steps = np.copy(nsteps)
        print(
            "Monitoring autocorrelation time every ",
            xtra_steps,
            " steps up to a maximum of ",
            max_n,
        )
        # Track how the average autocorrelation time estimate changes and test convergence
        step_count = [steps]
        autocorr = []
        old_tau = np.inf
        index = 0
        converged = False
        loc_idxs = []
        n = 0
        for i in range(9):
            loc_idxs.append(3 + n)
            loc_idxs.append(4 + n)
            loc_idxs.append(5 + n)
            n += 10
        print("location idxs: ", loc_idxs)
        while converged == False:
            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0, quiet=True)
            print("tau: ", tau)
            print("loc_tau: ", tau[loc_idxs])
            autocorr.append(np.mean(tau))
            # Check convergence
            converged = np.all(tau[loc_idxs] * 50 < steps)
            print("Steps taken: ", steps)
            print("First check: loc_tau*50 < steps: ", converged)
            # converged &= np.all(np.abs(old_tau - tau) / tau < 0.05)
            # print('Second check: all changed by less than 5% ', converged)
            index += 1
            if converged:
                plot_acorr(autocorr, step_count)
                break

            if steps > max_n:
                plot_acorr(autocorr, step_count)
                print(
                    "max steps reached withut convergence criterion being met. Continuing..."
                )
                raise ValueError(
                    "Number of mcmc steps exceeds maximum number set of ", max_n
                )

            old_tau = tau
            # if index == 1:
            #    steps += 10000
            #    step_count.append(steps)
            #    pos, lnp,  _ = sampler.run_mcmc(pos, 10000, progress=True)

            steps += xtra_steps
            step_count.append(steps)
            pos, lnp, _ = sampler.run_mcmc(pos, xtra_steps, progress=True)

        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))
        ac = np.mean(tau)
        print("Setting self.acorr to ", ac)

        return sampler, pos, ac, steps

    end = time.time()
    multi_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))

    return sampler, pos


def plot_acorr(autocorr, step_count):
    plt.figure(figsize=(10, 7))
    n = np.asarray(step_count)
    y = np.asarray(autocorr)
    plt.plot(n, n / 50.0, "--k")
    plt.plot(n, y)
    plt.xlim(0, n.max() * 1.05)
    plt.ylim(0, y.max() * 1.05)
    plt.xlabel("number of steps")
    plt.ylabel(r"mean $\hat{\tau}$")
    plt.savefig("auto_corr_plot.pdf")


class BF_Fit(object):
    def __init__(
        self,
        fname,
        system,
        rvstd,
        nspec,
        par_per_BF,
        n_indiv_pars,
        ln_wn,
        ln_var,
        ln_ls,
        nCPUs,
        burn_and_resample,
        nsteps,
        nburn,
        nwalkers,
        preburn1,
        preburn2,
        acorr,
        run,
        do_savefig,
        do_plot,
        do_chains,
        do_cornerplots,
        monitor_acorr,
    ):
        self.fname = fname
        self.system = system
        self.rvstd = rvstd
        self.nspec = nspec
        self.par_per_BF = par_per_BF
        self.n_indiv_pars = n_indiv_pars
        self.data = self.dict_of_dicts(
            ["bfsmoothlist", "rv_axis", "barycorlistnew", "normfits", "rv_corrections"]
        )
        self.par = self.dict_of_dicts(["names", "par", "medians", "max_like"])
        self.initial_array = np.empty([self.nspec, self.par_per_BF])
        self.ln_wn = ln_wn
        self.ln_var = ln_var
        self.ln_ls = ln_ls
        self.nCPUs = nCPUs
        self.burn_and_resample = burn_and_resample
        self.preburn1 = preburn1
        self.preburn2 = preburn2
        self.acorr = acorr
        self.run = run
        self.do_savefig = do_savefig
        self.do_plot = do_plot
        self.do_chains = do_chains
        self.do_cornerplots = do_cornerplots
        self.monitor_acorr = monitor_acorr
        if self.acorr < 500:
            print(
                "\n\t acorr set to ", self.acorr, " Set to 500 for final modelling? \n"
            )
        self.model = self.dict_of_dicts(["chain", "samples", "lnlike", "acc_frac"])
        self.model["nsteps"] = nsteps
        self.model["nburn"] = nburn
        self.model["nwalkers"] = nwalkers
        self.model["nsteps_total"] = nsteps
        self.logfile = "_BF_Fit_logfile_1.log"
        if os.path.isfile(self.logfile):
            logfiles = glob.glob("_BF_Fit_logfile_*.log")
            number = max(
                [
                    int(re.search("_BF_Fit_logfile_(.*).log", fname).group(1))
                    for fname in logfiles
                ]
            )
            self.logfile = "_BF_Fit_logfile_{:}.log".format(int(number) + 1)
        f = open(self.logfile, "w").close()

    def get_chains(self):
        print("nsteps total = ", self.model["nsteps_total"])
        nsteps = np.shape(self.model["chain"])[1]
        nburn = int(
            float(self.model["nburn"]) / float(self.model["nsteps_total"]) * nsteps
        )
        pname = "{0:s}/{1:s}_{2:s}_chains_{3:s}.png".format(
            self.plotdir, self.system, self.date_time, self.save_info
        )
        # print(pname)
        # raise ValueError('Number of spectra in array different to given nspec')
        chains.plot_chains(
            self.model["chain"],
            self.model["lnlike"],
            nsteps,
            self.model["nwalkers"],
            nburn,
            self.model["npar"],
            self.par["names"],
            50,
            pname,
            self.do_savefig,
        )

    def setup_call(self):
        self.create_logger()
        self.read_data()
        self.set_initial_params()
        self.log_inputs()
        self.create_directories()

    def mcmc_call(self):
        if self.run:
            self.setup_call()
            self.mcmc()
            self.get_acorr()
            self.get_model_results()
            self.print_mcmc_stats()
            self.get_param_names()
            self.get_save_info()
            self.get_posterior_parameters()
            if self.do_chains == True:
                self.get_chains()
            if self.do_plot == True:
                self.plotGP_offset()
            if self.do_cornerplots == True:
                self.corner_plots()
            self.move_files()

    def log_inputs(self):
        self.logger.info("\n\t INPUT PARAMETERS: ")
        self.logger.info("\n\t fname = {0}".format(self.fname))
        self.logger.info("\t system  = {0}".format(self.system))
        self.logger.info("\t rvstd  = {0}".format(self.rvstd))
        self.logger.info("\t nspec  = {0}".format(self.nspec))
        self.logger.info("\t npar  = {0}".format(self.model["npar"]))
        self.logger.info("\t par per bf  = {0}".format(self.par_per_BF))
        self.logger.info("\t n_indiv_pars  = {0}".format(self.n_indiv_pars))
        self.logger.info("\t ln_white_noise  = {0}".format(self.ln_wn))
        self.logger.info("\t ln_length_scale  = {0}".format(self.ln_ls))
        self.logger.info("\t nCPUs = {0}".format(self.nCPUs))
        self.logger.info("\t burn_and_resample  = {0}".format(self.burn_and_resample))
        self.logger.info("\t preburn1  = {0}".format(self.preburn1))
        self.logger.info("\t preburn2  = {0}".format(self.preburn2))
        self.logger.info("\t nsteps = {0}".format(self.model["nsteps"]))
        self.logger.info("\t nburn  = {0}".format(self.model["nburn"]))
        self.logger.info("\t nwalkers  = {0}".format(self.model["nwalkers"]))
        self.logger.info("\t acorr  = {0}".format(self.acorr))
        self.logger.info("\t do_savefig = {0}".format(self.do_savefig))
        self.logger.info("\t do_plot  = {0}".format(self.do_plot))
        self.logger.info("\t do_chains  = {0}".format(self.do_chains))
        self.logger.info("\t do_cornerplots  = {0}".format(self.do_cornerplots))
        self.logger.info("\t monitor_acorr  = {0}".format(self.monitor_acorr))
        self.logger.info("\t run  = {0}".format(self.run))

    def read_data(self):
        with fits.open(self.fname) as hdul:
            self.data["bfsmoothlist"] = hdul["bfsmoothlist"].data
            self.data["rv_axis"] = hdul["rv_axis"].data
            self.data["barycorlistnew"] = hdul["barycorlistnew"].data
            self.data["normfits"] = hdul["normfits"].data
            if self.nspec != len(self.data["normfits"]):
                raise ValueError("Number of spectra in array different to given nspec")
        # Shift rv axis into barycentric frame for each observation
        self.RV = []
        self.data["rv_corrections"] = []

        for i in range(1, len(self.data["barycorlistnew"])):
            self.data["rv_corrections"].append(
                self.data["barycorlistnew"][i]
                - self.data["barycorlistnew"][0]
                + self.rvstd
            )
            self.RV.append(self.data["rv_axis"] + self.data["rv_corrections"][i - 1])

        # Broadening function data for each spectrum
        self.BF = self.data["bfsmoothlist"][1:]

    def set_initial_params(self):
        # Create list of dictionaries of Gaussian amplitudes, locations, ln(widths^2) and ln(GP Kernal params) for each spectrum
        gausspars = []
        for i in range(self.nspec):
            est = dict(
                amp_1=self.data["normfits"][i][0],
                loc_1=self.data["normfits"][i][1] + self.data["rv_corrections"][i],
                log_sig2_1=np.log(self.data["normfits"][i][2] ** 2),
                amp_2=self.data["normfits"][i][3],
                loc_2=self.data["normfits"][i][4] + self.data["rv_corrections"][i],
                log_sig2_2=np.log(self.data["normfits"][i][5] ** 2),
                amp_3=self.data["normfits"][i][6],
                loc_3=self.data["normfits"][i][7] + self.data["rv_corrections"][i],
                log_sig2_3=np.log(self.data["normfits"][i][8] ** 2),
            )
            gausspars.append(est)

        # Create array of initial parameters in sequence N*(per spectrum individual params), joint params:
        # N*(amp1, amp2, amp3, loc1, loc2), logsig2_1, logsig2_2, logsig2_3, loc3, logwn, logsig_var, loglen_scale
        pars = []
        # Individual params:
        print("\n\t Individual parameters are: amp1,amp2,amp3,loc1,loc2,loc3,offset \n")
        for par in gausspars:
            pars.append(par.get("amp_1"))
            pars.append(par.get("amp_2"))
            pars.append(par.get("amp_3"))
            pars.append(par.get("loc_1"))
            pars.append(par.get("loc_2"))
            pars.append(par.get("loc_3"))
            pars.append(0.0)  # zero point offset
            pars.append(self.ln_var)  # log signal var  for GP
            pars.append(self.ln_ls)  # log length scale  for GP
            pars.append(self.ln_wn)  # log white_noise

        ##Check that zero point offset is at appropriate index:
        if pars[self.n_indiv_pars - 4] != 0.0:
            raise ValueError(
                "check that zero point offset parameter is at incorrect index."
            )

        # Get averages of joint params:
        avg_width1 = []
        avg_width2 = []
        avg_width3 = []
        # avg_loc3 = []

        for par in gausspars:
            avg_width1.append(par.get("log_sig2_1"))
            avg_width2.append(par.get("log_sig2_2"))
            avg_width3.append(par.get("log_sig2_3"))
            # avg_loc3.append(par.get('loc_3'))

        # Last 2 BFs have peak 3 blended, so values unreliable
        avg_width3.pop(-1)
        avg_width3.pop(-1)
        # avg_loc3.pop(-1)
        # avg_loc3.pop(-1)

        avg_width1 = np.mean(avg_width1)
        avg_width2 = np.mean(avg_width2)
        avg_width3 = np.mean(avg_width3)
        # avg_loc3 = np.mean(avg_loc3)

        pars.append(avg_width1)
        pars.append(avg_width2)
        pars.append(avg_width3)
        # pars.append(avg_loc3)

        print(pars)
        bf9loc3idx = 5 + (self.nspec - 1) * self.n_indiv_pars
        print("original bf9_loc3 guess = ", pars[bf9loc3idx])
        pars[bf9loc3idx] = 6.0
        print("updated bf9_loc3 guess = ", pars[bf9loc3idx])
        print(pars)

        self.initial_pars_list = np.asarray(pars)
        self.model["npar"] = len(self.initial_pars_list)

        for j in range(self.nspec):
            self.initial_array[j][0 : self.n_indiv_pars] = pars[
                j * self.n_indiv_pars : j * self.n_indiv_pars + self.n_indiv_pars
            ]
            self.initial_array[j][self.n_indiv_pars :] = pars[
                self.nspec * self.n_indiv_pars :
            ]

    def mcmc(self):
        # setup backend for saving chain

        self.chainfile = "_BF_Fit_chain_1.h5"
        if os.path.isfile(self.chainfile):
            chainfiles = glob.glob("_BF_Fit_chain_*.h5")
            number = max(
                [
                    int(re.search("_BF_Fit_chain_(.*).h5", fname).group(1))
                    for fname in chainfiles
                ]
            )
            self.chainfile = "_BF_Fit_chain_{:}.h5".format(int(number) + 1)
        self.logger.info("\t Temporary chain file is: {0:}".format(self.chainfile))

        # set up and run the MCMC
        self.t_in_gpe = self.get_time("in", run="MCMC")
        self.logger.info("\t Running on {:d} CPUs".format(self.nCPUs))
        del self.logger

        if self.monitor_acorr == True:
            self.sampler, self.pos, self.acorr, self.model["nsteps"] = call_mcmc(
                self.initial_pars_list,
                self.model["nwalkers"],
                self.model["nsteps"],
                self.model["npar"],
                self.ln_prob,
                self.nCPUs,
                self.chainfile,
                self.burn_and_resample,
                self.monitor_acorr,
            )
        else:
            self.sampler, self.pos = call_mcmc(
                self.initial_pars_list,
                self.model["nwalkers"],
                self.model["nsteps"],
                self.model["npar"],
                self.ln_prob,
                self.nCPUs,
                self.chainfile,
                self.burn_and_resample,
                self.monitor_acorr,
            )

        self.create_logger()
        if self.monitor_acorr == True:
            self.model["nsteps_total"] = self.model["nsteps"]
            self.logger.info("\t new acorr  = {0}".format(self.acorr))
            self.logger.info("\t new nsteps  = {0}".format(self.model["nsteps"]))
            self.logger.info(
                "\t new nsteps total  = {0}".format(self.model["nsteps_total"])
            )

        self.get_time("out", self.t_in_gpe, "MCMC")

    def get_acorr(self):
        # get info from sampler
        self.model["acorr"] = 1
        if self.model["nsteps"] > 500:
            if self.acorr != None:
                self.logger.info(
                    "\n\t Setting autocorr to requested value of {0:} \n".format(
                        self.acorr
                    )
                )
                self.model["acorr"] = self.acorr
            else:
                try:
                    self.model["acorr"] = max(self.sampler.get_autocorr_time())
                except:
                    self.logger.info(
                        "\n\t Autocorr not computed successfully: setting to 100."
                    )
                    self.model["acorr"] = 100
                if not np.isfinite(self.model["acorr"]):
                    self.model["acorr"] = 100

    def get_model_results(self):
        # calculate statistics from MCMC run
        self.min_acc_frac = 5.0

        jump = int(math.ceil(self.model["acorr"]))
        sample_jump = int(self.model["nburn"] / float(jump))
        self.model["full_chain"] = self.sampler.chain
        self.model["acc_frac"] = self.sampler.acceptance_fraction * 100.0
        lacc = self.model["acc_frac"] > self.min_acc_frac
        self.model["chain"] = self.sampler.chain[lacc, ::jump, :]
        self.model["samples"] = self.model["chain"][:, sample_jump:, :].reshape(
            (-1, self.model["npar"])
        )
        self.model["full_lnlike"] = self.sampler.lnprobability
        self.model["lnlike"] = self.model["full_lnlike"][lacc, ::jump]
        self.model["lnlike4samples"] = self.model["lnlike"][:, sample_jump:].reshape(-1)
        self.model["full_flat_chain"] = self.sampler.get_chain(flat=True)
        self.model["nwalkers"] = np.sum(lacc)
        if self.run:
            del self.sampler

    def print_mcmc_stats(self):
        acc_med, acc_sig = self.medsig(self.model["acc_frac"])
        acc_mean, acc_std = np.nanmean(self.model["acc_frac"]), np.std(
            self.model["acc_frac"]
        )
        lnlike = self.model["full_lnlike"][:, self.model["nburn"] :].flatten()
        lnlike_med, lnlike_sig = self.medsig(lnlike)
        self.logger.info(
            "\n\t full_chain shape = {0}".format(np.shape(self.model["full_chain"]))
        )
        self.logger.info(
            "\t chain shape      = {0}".format(np.shape(self.model["chain"]))
        )
        self.logger.info(
            "\t full flat chain shape  = {0}".format(
                np.shape(self.model["full_flat_chain"])
            )
        )
        self.logger.info(
            "\t samples shape      = {0}".format(np.shape(self.model["samples"]))
        )
        self.logger.info(
            "\n\t ln(likelihood)        =  {0:.0f} +/- {1:.0f}".format(
                lnlike_med, lnlike_sig
            )
        )
        self.logger.info(
            "\t Median acceptance     =  {0:.0f} +/- {1:.0f}".format(acc_med, acc_sig)
        )
        self.logger.info(
            "\t Mean acceptance       =  {0:.0f} +/- {1:.0f}".format(acc_mean, acc_std)
        )
        self.logger.info(
            "\t Autocorrelation time  =  {:.0f} steps \n".format(self.model["acorr"])
        )
        self.model["lnlike_med"], self.model["lnlike_sig"] = lnlike_med, lnlike_sig
        self.model["acc_frac_med"], self.model["acc_frac_sig"] = acc_med, acc_sig
        self.model["acc_frac_mean"], self.model["acc_frac_std"] = acc_mean, acc_std
        if (acc_med == 0.0 and acc_mean == 0.0) and self.do_savefig:
            raise ValueError(
                "Median/mean acceptance is 0. Check initial guesses are not excluded by priors."
            )

    def gaussmodel(self, p, RV_i):
        (
            amp_1,
            amp_2,
            amp_3,
            loc_1,
            loc_2,
            loc_3,
            offset,
            _,
            _,
            _,
            log_sig2_1,
            log_sig2_2,
            log_sig2_3,
        ) = p
        g1 = amp_1 * np.exp(-0.5 * (RV_i - loc_1) ** 2 * np.exp(-log_sig2_1))
        g2 = amp_2 * np.exp(-0.5 * (RV_i - loc_2) ** 2 * np.exp(-log_sig2_2))
        g3 = amp_3 * np.exp(-0.5 * (RV_i - loc_3) ** 2 * np.exp(-log_sig2_3))
        return g1 + g2 + g3 + offset

    def ln_like(self, p, RV_i, BF_i):
        a, tau, wn = np.exp(p[-6:-3])
        # gp = george.GP(a * kernels.ExpSquaredKernel(tau), white_noise=np.log(wn), fit_white_noise=True)
        gp = george.GP(a * kernels.ExpSquaredKernel(tau))
        gp.compute(RV_i, yerr=wn)
        return gp.lnlikelihood(BF_i - self.gaussmodel(p, RV_i))

        # residuals = y/model(p, t) - 1
        # return gp.lnlikelihood(residuals)

    def ln_prior(self, p, initial_vals):
        (
            amp_1,
            amp_2,
            amp_3,
            loc_1,
            loc_2,
            loc_3,
            offset,
            lna,
            lntau,
            lnwn,
            log_sig2_1,
            log_sig2_2,
            log_sig2_3,
        ) = p
        a1, a2, a3, l1, l2, l3, off, la, lt, lw, ls1, ls2, ls3 = initial_vals

        if (
            0 < amp_1 < 2 * a1
            and 0 < amp_2 < 2 * a2
            and 0 < amp_3 < 2 * a3
            and l1 - 50 < loc_1 < l1 + 50
            and l2 - 50 < loc_2 < l2 + 50
            and -0.2 * a1 < offset < 0.2 * a1
            and ls1 - ls1 / 2 < log_sig2_1 < ls1 + ls1 / 2
            and ls2 - ls2 / 2 < log_sig2_2 < ls2 + ls2 / 2
            and ls3 - ls3 / 2 < log_sig2_3 < ls3 + ls3 / 2
            and l3 - 10 < loc_3 < l3 + 10
            and -40 < lnwn < -1
            and -40 < lna < 40
            and -40 < lntau < 40
        ):
            return 0.0

        return -np.inf

    def ln_prob(self, pars):
        loglike = 0
        pars_array = np.empty([self.nspec, self.par_per_BF])
        for j in range(self.nspec):
            pars_array[j][0 : self.n_indiv_pars] = pars[
                j * self.n_indiv_pars : j * self.n_indiv_pars + self.n_indiv_pars
            ]
            pars_array[j][self.n_indiv_pars :] = pars[self.nspec * self.n_indiv_pars :]
        for i in range(self.nspec):
            lp = self.ln_prior(pars_array[i], self.initial_array[i])
            lnlk = self.ln_like(pars_array[i], self.RV[i], self.BF[i])
            if np.isfinite(lp):
                loglike += lp + lnlk
            else:
                loglike += -np.inf
        return loglike

    def create_logger(self):
        self.logger = logging.getLogger("BF_Fit")

        while self.logger.handlers:
            self.logger.handlers.pop()

        if not len(self.logger.handlers):
            logFormatter = logging.Formatter()

            fileHandler = logging.FileHandler(self.logfile)
            fileHandler.setLevel(logging.DEBUG)
            fileHandler.setFormatter(logFormatter)

            consoleHandler = logging.StreamHandler(sys.stdout)
            consoleHandler.setLevel(logging.INFO)
            consoleHandler.setFormatter(logFormatter)

            self.logger.addHandler(fileHandler)
            self.logger.addHandler(consoleHandler)

            self.logger.setLevel(logging.DEBUG)
            self.logger.propagate = False

    def print_logfile(self):
        self.logger.info("\n\t Temporary log file is: {0:}".format(self.logfile))

    def get_time(self, call="in", t_in=None, run=""):
        if call == "in":
            self.logger.info(
                "\n\t %s started at %s\n" % (run, str(datetime.datetime.now())[11:-10])
            )
            return time.time()
        elif call == "out":
            mins, secs = divmod((time.time() - t_in), 60)
            self.logger.info(
                "\n\t %s finished at %s:  time taken  =  %02d:%02d mins"
                % (run, str(datetime.datetime.now())[11:-10], mins, secs)
            )

    def dict_of_dicts(self, keys):
        temp = {}
        for item in keys:
            temp[item] = {}
        return temp

    def dict_of_OrderedDicts(self, keys):
        temp = {}
        for item in keys:
            temp[item] = OrderedDict()
        return temp

    def medsig(self, a):
        """Compute median and MAD-estimated scatter of array a"""
        med = np.nanmedian(a)
        sig = 1.48 * np.nanmedian(np.abs(a - med))
        return med, sig

    def create_directories(self):
        datadir = os.path.dirname(self.fname)
        orig_data_dir = datadir + "/_original"
        file_dir = datadir.replace("_data", "_files")
        plot_dir = datadir.replace("_data", "_plots")
        file_test_dir = file_dir + "/_tests"
        plot_test_dir = plot_dir + "/_tests"

        dirs = [orig_data_dir, file_dir, plot_dir, file_test_dir, plot_test_dir]
        for d in dirs:
            if not os.path.isdir(d):
                os.makedirs(d)

        self.tests = "" if self.model["nsteps"] > 500 else "/_tests"
        self.plotdir = "_systems/{0:s}/_plots{1:s}".format(self.system, self.tests)
        self.filedir = "_systems/{0:s}/_files{1:s}".format(self.system, self.tests)

    def plotGP_offset(self):
        t = self.RV
        y = self.BF
        samples = self.model["samples"]

        xlist = np.empty([self.nspec, 1000])
        for k in range(self.nspec):
            x = np.linspace(min(t[k]) - 20, max(t[k]) + 20, 1000)
            xlist[k] = x

        # Create array with 50th percentile values from MCMC results
        mcmc = []
        for i in range(self.model["npar"]):
            mcmc.append(np.percentile(samples[:, i], 50))

        mcmc_array = np.empty([self.nspec, self.par_per_BF])
        for i in range(self.nspec):
            mcmc_array[i][0 : self.n_indiv_pars] = mcmc[
                i * self.n_indiv_pars : i * self.n_indiv_pars + self.n_indiv_pars
            ]
            mcmc_array[i][self.n_indiv_pars :] = mcmc[self.nspec * self.n_indiv_pars :]

        offset_idx = self.n_indiv_pars - 4

        # plt.rc('font', family='serif', serif='Times')
        plt.rc("text", usetex=True)
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        plt.rc("axes", labelsize=8)
        # Plot dimensions to save to:
        width = 5
        height = width / 1.618

        for i in range(self.nspec):
            fig, ax = plt.subplots()
            fig.subplots_adjust(left=0.15, bottom=0.16, right=0.99, top=0.97)

            # Compute the prediction conditioned on the observations minus model for 50th percentile values:
            a, tau, wn = np.exp(mcmc_array[i][-6:-3])
            # np.random.seed(123)
            gp = george.GP(a * kernels.ExpSquaredKernel(tau))
            gp.compute(t[i], yerr=wn)
            pred, pred_var = gp.predict(y[i], xlist[i], return_var=True)
            gpterm = gp.sample_conditional(
                y[i] - self.gaussmodel(mcmc_array[i], t[i]), xlist[i]
            )
            gaussterm = self.gaussmodel(mcmc_array[i], t[i]) - mcmc_array[i][offset_idx]
            plt.errorbar(
                t[i],
                y[i] - mcmc_array[i][offset_idx],
                fmt="xk",
                capsize=0,
                markersize=3,
                mew=0.25,
            )
            plt.plot(xlist[i], gpterm - 0.01, color="blue", linestyle="--", lw=0.5)
            plt.plot(t[i], gaussterm - 0.01, color="red", linestyle="-.", lw=0.5)
            plt.plot(xlist[i], pred - mcmc_array[i][offset_idx], "springgreen", lw=0.75)
            plt.fill_between(
                xlist[i],
                pred - np.sqrt(pred_var) - mcmc_array[i][offset_idx],
                pred + np.sqrt(pred_var) - mcmc_array[i][offset_idx],
                color="springgreen",
                alpha=0.2,
            )
            plt.ylim(-0.02, 0.05)
            ax.set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=8)
            ax.set_ylabel("Broadening Function", fontsize=8)
            ax.tick_params(
                direction="in",
                which="both",
                bottom=True,
                top=True,
                left=True,
                right=True,
                labelsize=8,
            )
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.tick_params(which="major", length=5, width=1)
            ax.tick_params(which="minor", length=3, width=0.66)
            fig.set_size_inches(width, height)
            # plt.show()
            plt.savefig(
                self.plotdir
                + "/{0:s}_{1:s}_gpfits_offset_{2:d}.pdf".format(
                    self.system, self.date_time, i + 1
                )
            )
            plt.close()

        fig = plt.figure(1, figsize=(18, 108))
        for p in range(1, self.nspec + 1):
            ax = fig.add_subplot(self.nspec, 1, p)
            i = p - 1
            a, tau, wn = np.exp(mcmc_array[i][-6:-3])
            # np.random.seed(123)
            gp = george.GP(a * kernels.ExpSquaredKernel(tau))
            gp.compute(t[i], yerr=wn)
            pred, pred_var = gp.predict(y[i], xlist[i], return_var=True)
            gpterm = gp.sample_conditional(
                y[i] - self.gaussmodel(mcmc_array[i], t[i]), xlist[i]
            )
            gaussterm = self.gaussmodel(mcmc_array[i], t[i]) - mcmc_array[i][offset_idx]
            plt.errorbar(t[i], y[i] - mcmc_array[i][offset_idx], fmt="xk", capsize=0)
            plt.plot(xlist[i], gpterm - 0.01, color="blue", linestyle="--")
            plt.plot(t[i], gaussterm - 0.01, color="red", linestyle="-.")
            plt.plot(xlist[i], pred - mcmc_array[i][offset_idx], "springgreen", lw=1.5)
            plt.fill_between(
                xlist[i],
                pred - np.sqrt(pred_var) - mcmc_array[i][offset_idx],
                pred + np.sqrt(pred_var) - mcmc_array[i][offset_idx],
                color="springgreen",
                alpha=0.2,
            )
            plt.ylim(-0.02, 0.05)
            ax.set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=15)
            ax.set_ylabel("Broadening Function", fontsize=15)
            ax.tick_params(
                direction="in",
                which="both",
                bottom=True,
                top=True,
                left=True,
                right=True,
                labelsize=15,
            )
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.tick_params(which="major", length=13, width=1.5)
            ax.tick_params(which="minor", length=7, width=1)
        # plt.show()
        plt.savefig(
            self.plotdir
            + "/{0:s}_{1:s}_gpfits_offset.png".format(self.system, self.date_time)
        )
        plt.close()

        fig = plt.figure(1, figsize=(18, 108))
        for p in range(1, self.nspec + 1):
            ax = fig.add_subplot(self.nspec, 1, p)
            i = p - 1
            a, tau, wn = np.exp(mcmc_array[i][-6:-3])
            # np.random.seed(123)
            gp = george.GP(a * kernels.ExpSquaredKernel(tau))
            gp.compute(t[i], yerr=wn)
            pred, pred_var = gp.predict(y[i], xlist[i], return_var=True)
            gpterm = gp.sample_conditional(
                y[i] - self.gaussmodel(mcmc_array[i], t[i]), xlist[i]
            )
            gaussterm = self.gaussmodel(mcmc_array[i], t[i]) - mcmc_array[i][offset_idx]
            plt.errorbar(t[i], y[i] - mcmc_array[i][offset_idx], fmt="xk", capsize=0)
            plt.plot(xlist[i], gpterm, color="blue", linestyle="--")
            plt.plot(t[i], gaussterm, color="red", linestyle="-.")
            plt.plot(xlist[i], pred - mcmc_array[i][offset_idx], "springgreen", lw=1.5)
            plt.fill_between(
                xlist[i],
                pred - np.sqrt(pred_var) - mcmc_array[i][offset_idx],
                pred + np.sqrt(pred_var) - mcmc_array[i][offset_idx],
                color="springgreen",
                alpha=0.2,
            )
            plt.ylim(-0.01, 0.05)
            ax.set_xlabel("Radial Velocity [km s$^{-1}$]", fontsize=15)
            ax.set_ylabel("Broadening Function", fontsize=15)
            ax.tick_params(
                direction="in",
                which="both",
                bottom=True,
                top=True,
                left=True,
                right=True,
                labelsize=15,
            )
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.tick_params(which="major", length=13, width=1.5)
            ax.tick_params(which="minor", length=7, width=1)
        # plt.show()
        plt.savefig(
            self.plotdir + "/{0:s}_{1:s}_gpfits.png".format(self.system, self.date_time)
        )
        plt.close()

    def get_save_info(self):
        # get date and time of run finishing
        dt = str(datetime.datetime.now())
        date = dt[:10].replace("-", "")
        time = dt[11:-10].replace(":", ".")
        self.date_time = "%s_%s" % (date, time)

        # get run info (MCMC and constraints)
        self.save_info = "nsteps_{0:d}_nwalkers_{1:d}_nburn_{2:d}".format(
            self.model["nsteps_total"], self.model["nwalkers"], self.model["nburn"]
        )

    def move_files(self):
        if self.run:
            shutil.copyfile(
                self.chainfile,
                "{0:s}/{1:s}_{2:s}_.h5".format(
                    self.filedir, self.system, self.date_time
                ),
            )
            os.remove(self.chainfile)
            shutil.copyfile(
                self.logfile,
                "{0:s}/{1:s}_{2:s}_.log".format(
                    self.filedir, self.system, self.date_time
                ),
            )
            os.remove(self.logfile)

    def get_param_names(self):
        self.par["names"] = []
        labels = [
            "amp1",
            "amp2",
            "amp3",
            "loc1",
            "loc2",
            "loc3",
            "offset",
            "logsignalvar",
            "loglenscale",
            "logwhitenoise",
            "logsig21",
            "logsig22",
            "logsig23",
        ]
        for i in range(self.nspec):
            for j in range(self.n_indiv_pars):
                self.par["names"].append("bf" + str(i + 1) + labels[j])
        for k in range(self.n_indiv_pars, self.par_per_BF):
            self.par["names"].append(labels[k])

    def get_posterior_parameters(self):
        # calculate the parameter medians and 1-sigma uncertainties
        self.par["par"] = self.get_percentiles_ndarray(self.model["samples"].T)

        self.par["medians"] = self.par["par"].T[0]

        # print results
        self.logger.info("")
        for name, par, uerr, lerr in zip(
            self.par["names"],
            self.par["par"].T[0],
            self.par["par"].T[1],
            self.par["par"].T[2],
        ):
            par, uerr, lerr = self.round_sig_error2(par, uerr, lerr, n=2)
            # self.fmts.append( ".{:d}f".format(len(par.split('.')[1]) if '.' in par else 0) )

            pnspace = ["", " "] if "-" in par else [" ", ""]
            self.logger.info(
                "    {0:>{prec}s}   {1:s}{2:{prec1}s}{3:s}   [ + {4:{prec2}s}  - {5:{prec2}s} ]".format(
                    name,
                    pnspace[0],
                    par,
                    pnspace[1],
                    uerr,
                    lerr,
                    prec=len(max(self.par["names"])),
                    prec1=16,
                    prec2=12,
                )
            )

        self.par["max_like"] = self.model["samples"][
            np.argmax(self.model["lnlike4samples"])
        ]

        # My old method for sanity check:
        mcmc_results = []
        mcmc_results.append(
            ["parameter", "50th percentile value", "q50-q16", "q84-q50"]
        )
        for i in range(self.model["npar"]):
            mcmc = np.percentile(self.model["samples"][:, i], [15.87, 50, 84.14])
            q = np.diff(mcmc)
            mcmc_results.append([self.par["names"][i], mcmc[1], q[0], q[1]])

        with open(
            self.filedir
            + "/oldfmt_{0:s}_{1:s}_mcmc_results.txt".format(
                self.system, self.date_time
            ),
            "w",
        ) as f:
            for item in mcmc_results:
                f.write("%s\n" % item)
        # Also save samples:
        # np.savetxt(self.filedir+'/samples_{0:s}_{1:s}_.txt'.format(self.system, self.date_time), self.model['samples'])
        acorr_nburn_npar = np.array(
            [self.model["acorr"], self.model["nburn"], self.model["npar"]]
        )
        hdu1 = fits.PrimaryHDU()
        hdu2 = fits.ImageHDU(acorr_nburn_npar, name="acorr_nburn_npar")
        hdu3 = fits.ImageHDU(self.model["acc_frac"], name="acc_frac")
        new_hdul = fits.HDUList([hdu1, hdu2, hdu3])
        new_hdul.writeto(
            self.filedir
            + "/mcmc_acc_frac_{0:s}_{1:s}_.fits".format(self.system, self.date_time)
        )

    def get_percentiles_ndarray(self, xdistr):
        res = np.full((len(xdistr), 7), np.nan)
        for i, xd in enumerate(xdistr):
            if np.sum(np.abs(xd)) != 0 and all(np.isfinite(xd)):
                res[i] = self.get_percentiles(xd)
        # Return median and 1 sigma confidence intervals:
        return res[:, :3]

    def get_percentiles(self, xdistr):
        x = np.percentile(xdistr, [0.13, 2.28, 15.87, 50.0, 84.14, 97.72, 99.87])
        x = np.array(
            [
                x[3],
                x[4] - x[3],
                x[3] - x[2],
                x[5] - x[3],
                x[3] - x[1],
                x[6] - x[3],
                x[3] - x[0],
            ]
        )
        return x

    def round_sig(self, x, n):
        """round floating point x to n significant figures"""
        if isinstance(n, int) == False:
            raise TypeError("n must be an integer")
        try:
            x = float(x)
        except:
            raise TypeError("x must be a floating point object")
        form = "%0." + str(n - 1) + "e"
        st = form % x
        num, expo = epat.findall(st)[0]
        expo = int(expo)
        fs = str.split(num, ".")
        if len(fs) < 2:
            fs = [fs[0], ""]
        if expo == 0:
            return num
        elif expo > 0:
            if len(fs[1]) < expo:
                fs[1] += "0" * (expo - len(fs[1]))
            st = fs[0] + fs[1][0:expo]
            if len(fs[1][expo:]) > 0:
                st += "." + fs[1][expo:]
            return st
        else:
            expo = -expo
            if fs[0][0] == "-":
                fs[0] = fs[0][1:]
                sign = "-"
            else:
                sign = ""
            return sign + "0." + "0" * (expo - 1) + fs[0] + fs[1]

    def round_sig_error2(self, x, ex1, ex2, n):
        """
        Find min(ex1,ex2) rounded to n sig-figs, and make the floating
        point x and max(ex,ex2) match the number of decimals.
        """
        minerr = min(ex1, ex2)
        minstex = self.round_sig(minerr, n)
        if minstex.find(".") < 0:
            extra_zeros = len(minstex) - n
            sigfigs = len(str(int(x))) - extra_zeros
            if sigfigs < 1:
                sigfigs = 1  # len(str(minstex))
                stx = self.round_sig(x, sigfigs)
                maxstex = self.round_sig(max(ex1, ex2), sigfigs)
                stx = str(int(round(float(stx))))
                minstex = str(int(round(float(minstex))))
                maxstex = str(int(round(float(maxstex))))
            else:
                stx = self.round_sig(x, sigfigs)
                maxstex = self.round_sig(max(ex1, ex2), sigfigs)
        else:
            num_after_dec = len(str.split(minstex, ".")[1])
            stx = ("%%.%df" % num_after_dec) % (x)
            maxstex = ("%%.%df" % num_after_dec) % (max(ex1, ex2))
        if ex1 < ex2:
            return stx, minstex, maxstex
        else:
            return stx, maxstex, minstex

    def corner_plots(self):
        # Make corner plot
        labels = [
            "amp1",
            "amp2",
            "amp3",
            "loc1",
            "loc2",
            "loc3",
            "offset",
            "lnsigvar",
            "lnlenscale",
            "lnwnoise",
            r"$\ln sig^{2}_{1}$",
            r"$\ln sig^{2}_{2}$",
            r"$\ln sig^{2}_{3}$",
        ]

        samples = np.copy(self.model["samples"])
        truths = np.copy(self.par["medians"])

        n_joint_pars = self.par_per_BF - self.n_indiv_pars

        truths_joint = truths[-n_joint_pars:]
        truths_indiv = np.empty([self.nspec, self.n_indiv_pars])
        for j in range(self.nspec):
            truths_indiv[j][0 : self.n_indiv_pars] = truths[
                j * self.n_indiv_pars : j * self.n_indiv_pars + self.n_indiv_pars
            ]

        joint_samples = np.empty(
            [len(samples), n_joint_pars]
        )  # 'logsig2_1', 'logsig2_2', 'logsig2_3', 'loc3', 'logwhitenoise', 'log_signal_var', 'log_len_scale'
        indiv_samples = np.empty(
            [self.nspec, len(samples), self.n_indiv_pars]
        )  # ('amp1', 'amp2', 'amp3', 'loc1', 'loc2', 'offset')*N

        for j in range(len(samples)):
            joint_samples[j] = [
                samples[j][i]
                for i in list(
                    range(self.model["npar"] - n_joint_pars, self.model["npar"])
                )
            ]
            f = list(range(0, self.n_indiv_pars))
            for k in range(self.nspec):
                indiv_samples[k][j] = [samples[j][i] for i in f]
                f = [
                    f[i] + self.n_indiv_pars for i in list(range(0, self.n_indiv_pars))
                ]

        # Make corner plots
        fig = corner.corner(
            joint_samples,
            labels=labels[self.n_indiv_pars :],
            truths=truths_joint,
            quantiles=[0.1587, 0.5, 0.8414],
            show_titles=True,
            title_fmt=".4f",
            title_kwargs={"fontsize": 18},
            label_kwargs={"fontsize": 18},
            smooth=5,
            smooth1d=5,
            max_n_ticks=4,
        )
        size = self.par_per_BF
        fig.subplots_adjust(right=1.5, top=1.5)
        for ax in fig.get_axes():
            ax.tick_params(axis="both", labelsize=16)
        plt.gcf().set_size_inches(size, size)
        # plt.set_dpi(100)
        plt.savefig(
            self.plotdir
            + "/{0:s}_{1:s}_cornerplot_joint.png".format(self.system, self.date_time),
            bbox_inches="tight",
        )
        plt.close()

        for i in range(self.nspec):
            fig = corner.corner(
                indiv_samples[i],
                labels=labels[0 : self.n_indiv_pars],
                truths=truths_indiv[i],
                quantiles=[0.1587, 0.5, 0.8414],
                show_titles=True,
                title_fmt=".4f",
                title_kwargs={"fontsize": 18},
                label_kwargs={"fontsize": 18},
                smooth=5,
                smooth1d=5,
                max_n_ticks=4,
            )
            size = self.par_per_BF
            fig.subplots_adjust(right=1.5, top=1.5)
            for ax in fig.get_axes():
                ax.tick_params(axis="both", labelsize=16)
            plt.gcf().set_size_inches(size, size)
            plt.savefig(
                self.plotdir
                + "/{0:s}_{1:s}_cornerplot_indiv_{2:d}.png".format(
                    self.system, self.date_time, i + 1
                ),
                bbox_inches="tight",
            )
            plt.close()
