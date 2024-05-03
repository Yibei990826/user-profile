from dataclasses import dataclass
from functools import partial

import pandas as pd


@dataclass
class Cost:
    base_cost: float
    mid_cost: float
    high_cost: float


InputCost = Cost(base_cost=1.2, mid_cost=0.12, high_cost=0.06)
OutputCost = Cost(base_cost=3.2, mid_cost=0.32, high_cost=0.08)
FinetuneCost = Cost(base_cost=4.0, mid_cost=0.8, high_cost=0.4)


BASE_LIMIT = 1_000_000
MID_LIMIT = 10_000_000
TOKEN_COST_DIVISOR = 1_000  # Used for normalizing cost to per thousand tokens


def calculate_token_cost(tokens: int, costs: Cost, base_limit: int, mid_limit: int):
    base_cost = costs.base_cost
    mid_cost = costs.mid_cost
    high_cost = costs.high_cost
    # Calculate the cost of tokens based on tiered pricing
    base = min(tokens, base_limit) * base_cost / TOKEN_COST_DIVISOR
    mid = (
        max(0, min(tokens - base_limit, mid_limit - base_limit))
        * mid_cost
        / TOKEN_COST_DIVISOR
    )
    high = max(0, tokens - mid_limit) * high_cost / TOKEN_COST_DIVISOR
    total_cost = base + mid + high
    return pd.DataFrame(
        data={
            "cost_tier1": base,
            "cost_tier2": mid,
            "cost_tier3": high,
            "cost": total_cost,
        },
        index=[0],
    )


def apply_token_costs_to_dataframe(
    df,
    input_cost: Cost = InputCost,
    output_cost: Cost = OutputCost,
    finetune_cost: Cost = FinetuneCost,
    base_limit: int = BASE_LIMIT,
    mid_limit: int = MID_LIMIT,
):
    """
    Apply token cost calculation to a pandas DataFrame.

    :param df: DataFrame containing token usage data
    :return: DataFrame with additional columns for token costs
    """
    df = df.copy()
    calculate_token_fn = partial(
        calculate_token_cost,
        base_limit=base_limit,
        mid_limit=mid_limit,
    )

    def construct_costs_df(col, costs, suffix):
        df_cost = df[col].apply(
            calculate_token_fn,
            costs=costs,
        )
        df_cost = pd.concat(df_cost.values)
        for col in df_cost.columns:
            df[f"{col}_{suffix}"] = df_cost[col].values

    construct_costs_df("total_input_tokens", input_cost, "input")
    construct_costs_df("total_output_tokens", output_cost, "output")
    construct_costs_df("total_finetune_tokens", finetune_cost, "finetune")
    # Calculate total cost
    df["total_cost"] = df["cost_input"] + df["cost_output"] + df["cost_finetune"]

    # Filter rows where the total cost is greater than n
    # df = df[df['total_cost'] > 0.1]

    return df
