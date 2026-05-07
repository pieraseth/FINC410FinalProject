import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from scipy.optimize import brentq
import warnings
warnings.filterwarnings("ignore")

#  BLACK-SCHOLES PRICING

def bs_d1_d2(S, K, T, r, sigma):
    """Compute d1 and d2 for Black-Scholes."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def bs_price(S, K, T, r, sigma, option_type="put"):
    """Black-Scholes option price."""
    d1, d2 = bs_d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_greeks(S, K, T, r, sigma, option_type="put"):
    """
    Returns dict of Greeks:
      delta, gamma, theta (per calendar day), vega (per 1% vol), rho (per 1% rate)
    """
    d1, d2 = bs_d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega  = S * pdf_d1 * np.sqrt(T) / 100          # per 1% change in vol

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {"delta": delta, "gamma": gamma,
            "theta": theta, "vega": vega, "rho": rho}

def implied_vol(market_price, S, K, T, r, option_type="put"):
    """
    Solve for implied volatility using Brent's method.
    Returns IV or NaN if solution not found.
    """
    try:
        f = lambda sigma: bs_price(S, K, T, r, sigma, option_type) - market_price
        return brentq(f, 1e-6, 10.0, xtol=1e-8)
    except Exception:
        return np.nan


#  LOAD & PROCESS XOP PRICE DATA


def load_xop_prices(filepath):
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)
    df = df.sort_values("Date").reset_index(drop=True)
    df["Return"] = df["PX_LAST"].pct_change()
    return df

def calc_realized_vol(df, start="2026-03-03", end="2026-03-31", annualize=252):
    """Annualized realized volatility over a date window."""
    mask = (df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))
    rets = df.loc[mask, "Return"].dropna()
    return rets.std() * np.sqrt(annualize)

#  PART 3 — SIMPLE OPTIONS HEDGE (ATM PUT)

def simple_options_hedge(S0, K, T, r, sigma_iv, option_price_market,
                         portfolio_value, option_type="put",
                         contracts_multiplier=100):
    """
    S0              : XOP spot price on 2/27/2026
    K               : Strike price chosen
    T               : Time to expiry in years (e.g. 49/365 for Apr 17 expiry)
    r               : Risk-free rate (decimal, e.g. 0.0443)
    sigma_iv        : Implied vol (decimal) — from Bloomberg OMON
    option_price_market : Mid price of option from Bloomberg
    portfolio_value : $10,000,000
    contracts_multiplier: 100 shares per contract (standard equity options)
    """
    print("\n" + "="*60)
    print("PART 3: SIMPLE OPTIONS HEDGE — ATM PUT")
    print("="*60)

    # Greeks
    greeks = bs_greeks(S0, K, T, r, sigma_iv, option_type)
    print(f"\nOption Parameters:")
    print(f"  Spot (S0)         : ${S0:.2f}")
    print(f"  Strike (K)        : ${K:.2f}")
    print(f"  Days to Expiry    : {T*365:.0f}")
    print(f"  Impl. Vol (σ)     : {sigma_iv*100:.2f}%")
    print(f"  Risk-Free Rate    : {r*100:.2f}%")
    print(f"  Option Price (BS) : ${bs_price(S0, K, T, r, sigma_iv, option_type):.4f}")
    print(f"  Option Price (mkt): ${option_price_market:.4f}")

    print(f"\nGreeks:")
    for g, v in greeks.items():
        print(f"  {g.capitalize():8s}: {v:.6f}")

    # Delta-neutral hedge sizing
    # Number of shares in portfolio
    shares_held = portfolio_value / S0
    # Contracts needed: -portfolio_delta / (delta_per_contract)
    # delta_per_contract = delta * 100
    portfolio_delta = shares_held  # long stock → delta = shares
    delta_per_contract = greeks["delta"] * contracts_multiplier
    n_contracts = -portfolio_delta / delta_per_contract
    n_contracts_rounded = round(n_contracts)

    print(f"\nHedge Sizing (Delta-Neutral):")
    print(f"  Shares held       : {shares_held:,.0f}")
    print(f"  Portfolio delta   : {portfolio_delta:,.2f}")
    print(f"  Delta/contract    : {delta_per_contract:.4f}")
    print(f"  Contracts needed  : {n_contracts:.1f}  →  {n_contracts_rounded} contracts")

    # Hedge cost
    hedge_cost = n_contracts_rounded * option_price_market * contracts_multiplier
    print(f"\nHedge Cost:")
    print(f"  Premium paid      : ${hedge_cost:,.2f}")
    print(f"  Cost as % of port : {hedge_cost/portfolio_value*100:.3f}%")

    # Break-even
    if option_type == "put":
        breakeven = K - option_price_market
        print(f"\nBreak-even at expiry: ${breakeven:.2f}  "
              f"({(breakeven/S0 - 1)*100:.2f}% from spot)")
    else:
        breakeven = K + option_price_market
        print(f"\nBreak-even at expiry: ${breakeven:.2f}  "
              f"({(breakeven/S0 - 1)*100:.2f}% from spot)")

    # Implied vol vs realized vol — placeholder (fill after March data)
    print(f"\nVol Comparison (fill after March):")
    print(f"  Implied vol at initiation: {sigma_iv*100:.2f}%")
    print(f"  Realized vol in March    : [calculate with calc_realized_vol()]")

    return {
        "n_contracts": n_contracts_rounded,
        "hedge_cost": hedge_cost,
        "breakeven": breakeven,
        "greeks": greeks,
    }

# ─────────────────────────────────────────────
#  PART 4 — ADVANCED OPTIONS HEDGE (COLLAR)
# ─────────────────────────────────────────────

def advanced_options_hedge(S0, K_put, K_call, T, r,
                           sigma_put, sigma_call,
                           put_price_mkt, call_price_mkt,
                           portfolio_value, contracts_multiplier=100):
    """
    Collar strategy: Buy puts (downside protection) + Sell calls (finance premium).

    S0            : Spot price 2/27/2026
    K_put         : Put strike (e.g. 5% OTM below spot)
    K_call        : Call strike (e.g. 5-10% OTM above spot)
    T             : Time to expiry (years)
    r             : Risk-free rate
    sigma_put/call: IVs for each leg (from Bloomberg)
    put_price_mkt / call_price_mkt: Market mid prices
    """
    print("\n" + "="*60)
    print("PART 4: ADVANCED OPTIONS HEDGE — COLLAR")
    print("="*60)
    print(f"Strategy: Long {K_put:.2f} Put / Short {K_call:.2f} Call")
    print(f"Rationale: Limits downside below ${K_put:.2f} while capping upside "
          f"at ${K_call:.2f}. Net premium cost is reduced (or zero-cost if collar "
          f"is structured so call premium ≈ put premium).\n")

    # Greeks for each leg
    put_greeks  = bs_greeks(S0, K_put,  T, r, sigma_put,  "put")
    call_greeks = bs_greeks(S0, K_call, T, r, sigma_call, "call")

    print("Put leg Greeks:")
    for g, v in put_greeks.items():
        print(f"  {g.capitalize():8s}: {v:.6f}")

    print("\nCall leg Greeks (per contract, short → signs flip for the position):")
    for g, v in call_greeks.items():
        print(f"  {g.capitalize():8s}: {-v:.6f}  (short)")

    # Net collar greeks per contract pair
    net_delta_per_pair = put_greeks["delta"] - call_greeks["delta"]
    print(f"\nNet delta per contract pair (long put + short call): {net_delta_per_pair:.4f}")

    # Delta-neutral sizing
    shares_held = portfolio_value / S0
    portfolio_delta = shares_held
    net_delta_per_contract_pair = net_delta_per_pair * contracts_multiplier
    n_pairs = -portfolio_delta / net_delta_per_contract_pair
    n_pairs_rounded = round(n_pairs)

    print(f"\nHedge Sizing:")
    print(f"  Contract pairs needed: {n_pairs:.1f}  →  {n_pairs_rounded} pairs")

    # Collar cost
    net_premium_per_contract = put_price_mkt - call_price_mkt  # long put, short call
    total_cost = n_pairs_rounded * net_premium_per_contract * contracts_multiplier
    print(f"\nHedge Cost:")
    print(f"  Put premium  (paid)     : ${put_price_mkt:.4f}")
    print(f"  Call premium (received) : ${call_price_mkt:.4f}")
    print(f"  Net premium per share   : ${net_premium_per_contract:.4f}")
    print(f"  Total net cost          : ${total_cost:,.2f}  "
          f"({'paid' if total_cost > 0 else 'received'})")
    print(f"  Cost as % of portfolio  : {total_cost/portfolio_value*100:.3f}%")

    # Break-evens
    be_down = K_put  - net_premium_per_contract  # effective floor
    be_up   = K_call - net_premium_per_contract  # effective cap
    print(f"\nEffective floor : ${be_down:.2f}  ({(be_down/S0-1)*100:.2f}% from spot)")
    print(f"Effective cap   : ${be_up:.2f}   ({(be_up/S0-1)*100:.2f}% from spot)")

    return {
        "n_pairs": n_pairs_rounded,
        "total_cost": total_cost,
        "net_premium": net_premium_per_contract,
        "put_greeks": put_greeks,
        "call_greeks": call_greeks,
    }

# ─────────────────────────────────────────────
#  PnL BACKTEST
# ─────────────────────────────────────────────

def backtest_pnl(xop_df, S0,
                 # Simple put params
                 K_put_simple, T0_simple, r, sigma_iv_simple, n_contracts_simple,
                 # Collar params
                 K_put_col, K_call_col, T0_col, sigma_put_col, sigma_call_col,
                 n_pairs,
                 portfolio_value, contracts_multiplier=100,
                 start="2026-03-03", end="2026-03-31"):
    """
    Daily mark-to-market PnL for:
      1. Unhedged ETF position
      2. ETF + simple put hedge
      3. ETF + collar hedge
    """
    mask = (xop_df["Date"] >= pd.Timestamp(start)) & \
           (xop_df["Date"] <= pd.Timestamp(end))
    df = xop_df.loc[mask].copy().reset_index(drop=True)

    shares = portfolio_value / S0
    results = []

    for _, row in df.iterrows():
        S = row["PX_LAST"]
        date = row["Date"]
        # Time remaining (rough — from initiation date)
        T_remaining_simple = max((T0_simple - (date - pd.Timestamp("2026-02-27")).days / 365), 1e-6)
        T_remaining_col    = max((T0_col    - (date - pd.Timestamp("2026-02-27")).days / 365), 1e-6)

        # ETF PnL
        etf_pnl = (S - S0) * shares

        # Simple put PnL (mark option to model)
        put_val_simple = bs_price(S, K_put_simple, T_remaining_simple, r, sigma_iv_simple, "put")
        put_pnl_simple = (put_val_simple) * n_contracts_simple * contracts_multiplier

        # Collar PnL
        put_val_col  = bs_price(S, K_put_col,  T_remaining_col, r, sigma_put_col,  "put")
        call_val_col = bs_price(S, K_call_col, T_remaining_col, r, sigma_call_col, "call")
        # Long put, short call (we subtract initial premiums below for net cost)
        collar_pnl = (put_val_col - call_val_col) * n_pairs * contracts_multiplier

        results.append({
            "Date": date, "XOP": S,
            "ETF_PnL": etf_pnl,
            "Simple_Hedge_PnL": put_pnl_simple,
            "Collar_Hedge_PnL": collar_pnl,
            "Net_Simple": etf_pnl + put_pnl_simple,
            "Net_Collar": etf_pnl + collar_pnl,
        })

    return pd.DataFrame(results)

# ─────────────────────────────────────────────
#  PAYOFF DIAGRAMS
# ─────────────────────────────────────────────

def plot_payoff_diagrams(S0, K_put_simple, put_price,
                         K_put_col, K_call_col, net_premium_col,
                         portfolio_value, contracts_multiplier=100):
    """
    Payoff at expiry for:
      (a) Simple put hedge
      (b) Collar hedge
    Per-share basis for clarity; multiply by shares for full notional.
    """
    S_range = np.linspace(S0 * 0.70, S0 * 1.30, 500)
    shares = portfolio_value / S0

    # Simple put payoff
    put_payoff_simple = np.maximum(K_put_simple - S_range, 0) - put_price
    etf_pnl_range = (S_range - S0)
    net_simple = etf_pnl_range + put_payoff_simple

    # Collar payoff
    put_payoff_col  = np.maximum(K_put_col  - S_range, 0)
    call_payoff_col = np.maximum(S_range - K_call_col, 0)
    collar_payoff   = put_payoff_col - call_payoff_col - net_premium_col
    net_collar = etf_pnl_range + collar_payoff

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("XOP Hedging Strategies — Payoff at Expiry (per share)", fontsize=14)

    # --- Simple put ---
    ax = axes[0]
    ax.plot(S_range, etf_pnl_range, "b--", label="Unhedged ETF", alpha=0.6)
    ax.plot(S_range, net_simple,    "g-",  label="ETF + ATM Put", lw=2)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(S0,          color="gray",  ls=":", label=f"Spot ${S0:.2f}")
    ax.axvline(K_put_simple,color="red",   ls=":", label=f"Strike ${K_put_simple:.2f}")
    ax.fill_between(S_range, net_simple, 0, where=(net_simple < 0),
                    alpha=0.1, color="red", label="Loss zone")
    ax.fill_between(S_range, net_simple, 0, where=(net_simple > 0),
                    alpha=0.1, color="green", label="Gain zone")
    ax.set_title("Simple Hedge: Long ATM Put")
    ax.set_xlabel("XOP Price at Expiry")
    ax.set_ylabel("Profit / Loss per Share ($)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Collar ---
    ax = axes[1]
    ax.plot(S_range, etf_pnl_range, "b--", label="Unhedged ETF", alpha=0.6)
    ax.plot(S_range, net_collar,    "m-",  label="ETF + Collar", lw=2)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(S0,          color="gray",  ls=":", label=f"Spot ${S0:.2f}")
    ax.axvline(K_put_col,   color="red",   ls=":", label=f"Put Strike ${K_put_col:.2f}")
    ax.axvline(K_call_col,  color="orange",ls=":", label=f"Call Strike ${K_call_col:.2f}")
    ax.fill_between(S_range, net_collar, 0, where=(net_collar < 0),
                    alpha=0.1, color="red")
    ax.fill_between(S_range, net_collar, 0, where=(net_collar > 0),
                    alpha=0.1, color="green")
    ax.set_title("Advanced Hedge: Collar")
    ax.set_xlabel("XOP Price at Expiry")
    ax.set_ylabel("Profit / Loss per Share ($)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("xop_payoff_diagrams.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nPayoff diagram saved: xop_payoff_diagrams.png")

def plot_pnl_over_time(pnl_df):
    """Cumulative PnL over March for all three positions."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pnl_df["Date"], pnl_df["ETF_PnL"],   "b--", label="Unhedged ETF", lw=1.5)
    ax.plot(pnl_df["Date"], pnl_df["Net_Simple"], "g-",  label="ETF + Simple Put", lw=2)
    ax.plot(pnl_df["Date"], pnl_df["Net_Collar"], "m-",  label="ETF + Collar", lw=2)
    ax.axhline(0, color="black", lw=0.8)
    ax.fill_between(pnl_df["Date"], pnl_df["ETF_PnL"], 0,
                    where=(pnl_df["ETF_PnL"] < 0), alpha=0.08, color="red")
    ax.set_title("XOP Hedging Backtest — Daily Mark-to-Market PnL (March 2026)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("xop_pnl_backtest.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("PnL chart saved: xop_pnl_backtest.png")

# ─────────────────────────────────────────────
#  SUMMARY TABLE
# ─────────────────────────────────────────────

def print_summary_table(pnl_df, hedge_cost_simple, hedge_cost_collar):
    """Print a clean comparison table of hedge outcomes."""
    final = pnl_df.iloc[-1]
    print("\n" + "="*60)
    print("SUMMARY: HEDGE EFFECTIVENESS (Final PnL on 3/31/2026)")
    print("="*60)
    rows = [
        ("Unhedged ETF",   final["ETF_PnL"],   0,                 final["ETF_PnL"]),
        ("+ Simple Put",   final["ETF_PnL"],   hedge_cost_simple, final["Net_Simple"]),
        ("+ Collar",       final["ETF_PnL"],   hedge_cost_collar, final["Net_Collar"]),
    ]
    print(f"\n{'Strategy':<16} {'ETF PnL':>12} {'Hedge Cost':>12} {'Net PnL':>12}")
    print("-"*55)
    for name, etf, cost, net in rows:
        print(f"{name:<16} {etf:>12,.0f} {cost:>12,.0f} {net:>12,.0f}")

    print("\nHedge Effectiveness (% of unhedged loss eliminated):")
    unhedged_pnl = final["ETF_PnL"]
    if unhedged_pnl < 0:
        eff_simple = (final["Net_Simple"] - unhedged_pnl) / abs(unhedged_pnl) * 100
        eff_collar = (final["Net_Collar"] - unhedged_pnl) / abs(unhedged_pnl) * 100
        print(f"  Simple Put : {eff_simple:.1f}%")
        print(f"  Collar     : {eff_collar:.1f}%")
    else:
        print("  (ETF ended positive — hedges cost premium but downside protection unused)")

def main():
    xop_df = load_xop_prices("xop_prices.csv")

    S0 = 153.72
    r  = 0.0364
    T  = 49/365
    portfolio_value = 10_000_000

    # Part 3 - ATM Put
    K_put_simple        = 154.00
    sigma_iv_simple     = 0.3409
    option_price_simple = 7.90

    part3 = simple_options_hedge(
        S0=S0, K=K_put_simple, T=T, r=r,
        sigma_iv=sigma_iv_simple,
        option_price_market=option_price_simple,
        portfolio_value=portfolio_value
    )

    iv_check = implied_vol(option_price_simple, S0, K_put_simple, T, r, "put")
    print(f"\nIV cross-check (BS solve): {iv_check*100:.2f}%  "
          f"(Bloomberg gave: {sigma_iv_simple*100:.2f}%)")

    rv_march = calc_realized_vol(xop_df, "2026-03-03", "2026-03-31")
    print(f"\nMarch Realized Vol (annualized): {rv_march*100:.2f}%")
    print(f"Implied Vol at initiation:        {sigma_iv_simple*100:.2f}%")
    print(f"IV vs RV spread:                  {(sigma_iv_simple - rv_march)*100:.2f}% "
          f"({'options were rich' if sigma_iv_simple > rv_march else 'options were cheap'})")

    # Part 4 - Collar
    K_put_col      = 146.00
    K_call_col     = 160.00
    sigma_put_col  = 0.3451
    sigma_call_col = 0.3391
    put_price_col  = 4.40
    call_price_col = 4.98

    part4 = advanced_options_hedge(
        S0=S0, K_put=K_put_col, K_call=K_call_col,
        T=T, r=r,
        sigma_put=sigma_put_col, sigma_call=sigma_call_col,
        put_price_mkt=put_price_col, call_price_mkt=call_price_col,
        portfolio_value=portfolio_value
    )

    plot_payoff_diagrams(
        S0=S0,
        K_put_simple=K_put_simple, put_price=option_price_simple,
        K_put_col=K_put_col, K_call_col=K_call_col,
        net_premium_col=part4["net_premium"],
        portfolio_value=portfolio_value
    )

    pnl_df = backtest_pnl(
        xop_df=xop_df, S0=S0,
        K_put_simple=K_put_simple, T0_simple=T, r=r,
        sigma_iv_simple=sigma_iv_simple,
        n_contracts_simple=part3["n_contracts"],
        K_put_col=K_put_col, K_call_col=K_call_col,
        T0_col=T, sigma_put_col=sigma_put_col, sigma_call_col=sigma_call_col,
        n_pairs=part4["n_pairs"],
        portfolio_value=portfolio_value
    )

    print("\n--- Daily PnL Table ---")
    print(pnl_df[["Date","XOP","ETF_PnL","Net_Simple","Net_Collar"]].to_string(index=False))
    pnl_df.to_csv("xop_pnl_backtest.csv", index=False)

    plot_pnl_over_time(pnl_df)
    print_summary_table(pnl_df, part3["hedge_cost"], part4["total_cost"])


if __name__ == "__main__":
    main()