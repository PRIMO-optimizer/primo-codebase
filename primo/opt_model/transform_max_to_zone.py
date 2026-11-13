import pandas as pd
from primo.data_parser.default_data import WELL_BASED_METRICS, WELL_PAIR_METRICS


def modify_data_based_on_zones(model_inputs):
    wd = model_inputs.config.well_data
    eff_metrics = wd.config.efficiency_metrics

    for metric in eff_metrics.selected_metrics:
        binned_values = [
            (1 - eta) * getattr(model_inputs.config, f"max_{metric.name}")
            for eta in metric.zone_data.efficiency.values()
        ]

        if metric.name in WELL_BASED_METRICS:
            zone_values = pd.cut(
                wd[metric.data_col_name],
                bins=metric.zone_data.points,
                labels=binned_values,
                include_lowest=True,
            )
            wd.data[metric.data_col_name] = zone_values.astype(float)

        elif metric.name in WELL_PAIR_METRICS:
            for pair_data in model_inputs.pairwise_metrics.values():
                zone_values = pd.cut(
                    pair_data[metric.name],
                    bins=metric.zone_data.points,
                    labels=binned_values,
                    include_lowest=True,
                )
                pair_data[metric.name] = zone_values.astype(float)
