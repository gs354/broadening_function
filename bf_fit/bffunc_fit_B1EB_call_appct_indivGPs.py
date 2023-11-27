import bf_fit.bffunc_fit_B1EB_tertiary_acorr_indivGPs as x


x.call(
    fname="",
    system="B1EB",
    rvstd=30.568,
    nspec=9,
    par_per_BF=13,
    n_indiv_pars=10,
    ln_wn=-28.7,
    ln_var=-13.3,
    ln_ls=3.4,
    nCPUs=32,
    burn_and_resample=False,
    nsteps=200000,
    nburn=40000,
    nwalkers=500,
    preburn1=5000,
    preburn2=5000,
    acorr=3400,
    run=True,
    do_savefig=True,
    do_plot=True,
    do_chains=True,
    do_cornerplots=True,
    monitor_acorr=False,
)
