import bf_fit.bffunc_fit_binary_acorr_indivGPs_CELERITE2_SHOfixQ as x


x.call(
    fname="",
    system="_EB",
    rvstd=30.458,
    nspec=9,
    par_per_BF=10,
    n_indiv_pars=5,
    GP_kernel="SHO_fixQ",
    gp_pars_initial={"S0": 8.847e-5, "omega0": 0.13812},
    wn=4.6e-7,
    nCPUs=8,
    burn_and_resample=False,
    nsteps=60000,
    nburn=30000,
    nwalkers=400,
    preburn1=5000,
    preburn2=5000,
    acorr=1000,
    run=True,
    do_savefig=True,
    do_plot=True,
    do_chains=True,
    do_cornerplots=False,
    monitor_acorr=True,
)
