import matplotlib.pyplot as plt
import numpy as np


def plot_sizes(prefix, mc_repeats, out=None):
    plt.figure(figsize=(6, 6))
    deltas = np.load(f"{prefix}_deltas.npy")
    styles = ["-.", "--", "-"]
    for i, n in enumerate([100, 200, 400]):
        delta_rates = np.load(f"{prefix}_delta_rates_{n}.npy")
        delta_rates_std = np.load(f"{prefix}_delta_rates_std_{n}.npy")
        plt.plot(deltas, delta_rates, styles[i], linewidth=2, color="black")
        plt.fill_between(
            deltas,
            delta_rates - (delta_rates_std * 2.56 / np.sqrt(mc_repeats)),
            delta_rates + (delta_rates_std * 2.56 / np.sqrt(mc_repeats)),
            alpha=0.2,
            color="grey",
            label="95% CI",
        )

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.axhline(0.05, linestyle="--", color="black")
    plt.yticks(np.linspace(0, 1, 11))
    plt.ylim(0, 0.999)
    plt.xlim(deltas[0], deltas[-1])
    plt.xlabel(r"$\delta$")
    plt.ylabel(f"Rejection rate")
    plt.grid("on")

    if out:
        plt.savefig(
            out,
            bbox_inches="tight",
            transparent=True,
        )


def plot_power(prefix, mc_repeats, out=None):
    plt.figure(figsize=(6, 6))
    ns = np.load(f"{prefix}_ns.npy")

    local_power = np.load(f"{prefix}_local_power.npy")
    local_power_std = np.load(f"{prefix}_local_power_std.npy")

    plt.plot(ns, local_power, linewidth=2, color="black")
    plt.fill_between(
        ns,
        local_power - (local_power_std * 1.96 / np.sqrt(mc_repeats)),
        local_power + (local_power_std * 1.96 / np.sqrt(mc_repeats)),
        alpha=0.2,
        color="grey",
    )

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.yticks(np.linspace(0, 1, 11))
    plt.ylim(0, 0.999)
    plt.xlim(ns[0], ns[-1])
    plt.xlabel(r"$n$")
    plt.ylabel(f"Rejection rate")
    plt.grid("on")

    if out:
        plt.savefig(
            out,
            bbox_inches="tight",
            transparent=True,
        )


def plot_size(
    M,
    deltas,
    delta_rates,
    delta_rates_std,
):
    plt.figure(figsize=(6, 6))

    plt.plot(deltas, delta_rates, "-", linewidth=3, color="black")
    plt.fill_between(
        deltas,
        delta_rates - (delta_rates_std * 2.56 / np.sqrt(M)),
        delta_rates + (delta_rates_std * 2.56 / np.sqrt(M)),
        alpha=0.2,
        color="grey",
        label="95% CI",
    )

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.axhline(0.05, linestyle="--", color="black")
    plt.yticks(np.linspace(0, 1, 11))
    plt.ylim(0, 0.999)
    plt.xlim(deltas[0], deltas[-1])
    plt.xlabel(r"$\delta$")
    plt.ylabel(f"Rejection rate")
    plt.grid("on")


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
