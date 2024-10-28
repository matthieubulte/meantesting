import matplotlib.pyplot as plt
import numpy as np


def plot_sim_results(
    M,
    delta_n,
    deltas,
    delta_rates,
    delta_rates_std,
    local_ns,
    local_rates,
    local_rates_std,
):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)

    plt.errorbar(
        deltas,
        delta_rates,
        yerr=delta_rates_std * 1.96 / np.sqrt(M),
        fmt="o-",
        ecolor="red",
        capsize=5,
        label="Rejection ± 95% CI",
    )

    plt.axhline(0.05, color="r")
    plt.yticks(np.linspace(0, 1, 11))
    plt.xlabel("(Log_mu0 mu_delta) / d(mu0, mu1)")
    plt.ylabel(f"rejection rate n={delta_n}")
    plt.grid()

    plt.subplot(122)

    plt.errorbar(
        local_ns,
        local_rates,
        yerr=local_rates_std * 1.96 / np.sqrt(M),
        fmt="o-",
        ecolor="red",
        capsize=5,
        label="Rejection ± 95% CI",
    )

    plt.yticks(np.linspace(0, 1, 11))
    plt.xlabel("n")
    plt.ylabel("rejection rate vs mu=n^{-1/2}")
    plt.grid()
