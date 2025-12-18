# ecoagent/demo.py
from importlib import import_module
import time
import sys
import os
import textwrap
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from green_agent.green_agent import GreenAgent

def pause(seconds=5, msg=None):
    if msg:
        print(msg)
    total = seconds
    for i in range(total):
        sys.stdout.write(f"\r    Waiting {total - i:>2}s...")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\r" + " " * 24 + "\r")
    sys.stdout.flush()

def hr(char="─", width=60, label=None):
    line = char * width
    if label:
        pad = max(0, width - len(label) - 2)
        line = f"{label} {char * pad}"
    print(line)

def print_intro(ga: GreenAgent):
    info = ga.describe()

    print("=== Task Introduction ===")
    context = info["context"]
    target  = info["target"]
    horizon = info["horizon"]
    nmax    = info["budget"]["max_datasets"]
    guard   = info.get("leakage_guard_test_block", [])

    intro = f"""
    Task:
      We evaluate white agents on a resource-constrained forecasting task defined by the green agent.

    Environment:
      • Context: {context}
      • Target to forecast: {target}
      • Forecast horizon: {horizon} future point(s)
      • Dataset-access budget: {nmax} dataset(s) total

    Actions:
      • White agents may request datasets (each unique request counts toward budget).
      • White agents use requested historical data and output predictions for the test horizon.
      • The green agent scores accuracy (RMSE) and efficiency (unused budget bonus).
    """
    print(textwrap.dedent(intro).rstrip())

    input("\nPress [Enter] to see available datasets...")

    datasets = f"""
    Datasets:
      • Total housing units — Total Housing Units
      • Occupied units — Occupied Units
      • Vacant units — Vacant Units
      • Median listing price — Median Listing Price (US dollars)
      • Population — Total Population
      • Median age — Median Age (years)
      • Homeless total - Homeless Total
      • Median income — Median Household Income (inflation-adjusted US dollars)
      • Poverty rate — Percentage of Population at or below the Poverty Level

    Target: Vacant Units
    """

    print(textwrap.dedent(datasets).rstrip())

def run_agent(mod_name: str, title: str, context_dir="ecoagent/contexts/housing", pause_seconds=5):
    print(f"\n=== {title} ===")
    ga = GreenAgent(context_dir)
    pause(pause_seconds, msg=None)
    try:
        result = import_module(mod_name).run(ga)
        print(result)
    except Exception as e:
        print({"status": "error", "reason": str(e)})

def run_demo():
    ga = GreenAgent("ecoagent/contexts/housing")
    print_intro(ga)

    input("\nPress [Enter] to begin running white agents...")
    run_agent("white_agents.naive_last_value", "Agent 1 (naive last value)", pause_seconds=5)

    input("\nPress [Enter] to run next white agent...")
    run_agent("white_agents.two_feature_regression", "Agent 2 (two-feature regression)", pause_seconds=5)

    input("\nPress [Enter] to run next white agent...")
    run_agent("white_agents.over_budget", "Agent 3 (surpasses budget)", pause_seconds=5)

    print()
    print("=== Demo Complete ===")

if __name__ == "__main__":
    run_demo()