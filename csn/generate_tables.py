import os
import numpy as np
import pandas as pd


class TableGenerator:
    """Generates summary performance and specificity tables from automated music
    tagging triplet evaluation experiments."""

    def __init__(self, results_dir="runs-final", categories=["instrument", "genre", "mood"]):
        """Initializes the generator, loads all experiment CSVs, and constructs
        the wide-format result DataFrame."""

        self.results_dir = results_dir
        self.categories = categories
        self.df_all_wide = self.load_results()

    def load_results(self):
        """Loads triplet score CSV files across all backbone, pooling, variation,
        and masking configurations and pivots them into a unified wide DataFrame.

        What it does:
            Iterates through experiment directories, extracts Jamendo and MTT triplet
            scores for each run (5 runs per configuration), and pivots metric scores
            into `{dataset}_{metric}` columns per run.

        Output:
            pd.DataFrame: Wide-format DataFrame indexed by Backbone, Pooling, Variation,
            Learn Masks, Run, and Mask, with columns for each dataset metric
            (e.g., jamendo_instrument, mtt_genre).
        """
        records = []

        for pooling in ["attention", "mean"]:
            for backbone in ["muq", "clap", "mert"]:
                for variation in ["none", "one_layer", "mlp"]:
                    learn_masks_options = ([True] if variation == "none" else [True, False])

                    for learn_masks in learn_masks_options:
                        learn_masks_str = ("learned_mask" if learn_masks else "fixed_mask")

                        for run in range(5):
                            experiment_name = f"{backbone}_{pooling}_{variation}_{learn_masks_str}_run_{run}"
                            checkpoint_folder = os.path.join(self.results_dir, experiment_name)

                            save_jamendo = os.path.join(checkpoint_folder, "triplet_scores_jamendo.csv")
                            save_mtt = os.path.join(checkpoint_folder, "triplet_scores_mtt.csv")

                            if not (os.path.exists(save_jamendo) and os.path.exists(save_mtt)):
                                continue

                            df_jamendo = pd.read_csv(save_jamendo)
                            df_mtt = pd.read_csv(save_mtt)

                            if df_jamendo.index.name is None:
                                df_jamendo = df_jamendo.set_index(df_jamendo.columns[0])
                                df_mtt = df_mtt.set_index(df_mtt.columns[0])

                            for dataset_name, df_run in [("Jamendo", df_jamendo), ("MTT", df_mtt)]:
                                for mask_name in df_run.index:
                                    for col in df_run.columns:
                                        records.append(
                                            {
                                                "Dataset": dataset_name,
                                                "Backbone": backbone,
                                                "Pooling": pooling,
                                                "Variation": variation,
                                                "Learn Masks": learn_masks_str,
                                                "Run": run,
                                                "Mask": mask_name,
                                                "Metric": col,
                                                "value": df_run.loc[
                                                    mask_name, col
                                                ],
                                            }
                                        )

        df_all = pd.DataFrame(records)
        index_cols = ["Backbone", "Pooling", "Variation", "Learn Masks", "Run", "Mask"]

        df_wide = df_all.pivot(index=index_cols, columns=["Dataset", "Metric"], values="value")
        df_wide.columns = [f"{dataset.lower()}_{metric}" for dataset, metric in df_wide.columns]
        df_wide = df_wide.reset_index()
        df_wide.columns.name = None

        return df_wide

    def compute_specificity(self):
        """Computes specificity metrics across model configurations and datasets.

        What it does:
            For each model run, calculates attribute-level specificity defined as:
            $Specificity = OwnEffect - MeanCrossEffect$, where $OwnEffect$ is performance
            gain over unmasked backbone for the target attribute, and $MeanCrossEffect$
            is average performance gain on off-target attributes. Averages specificity
            across attributes per run, then computes mean and std across 5 runs per dataset.

        Output:
            pd.DataFrame: Columns for Backbone, Pooling, Variation, Learn Masks,
            and dataset-specific specificity aggregates (jamendo_specificity_mean,
            jamendo_specificity_std, mtt_specificity_mean, mtt_specificity_std).
        """
        config_cols = ["Backbone", "Pooling", "Variation", "Learn Masks", "Run"]
        datasets = ["jamendo", "mtt"]
        spec_records = []

        for config_vals, group in self.df_all_wide.groupby(config_cols):
            backbone, pooling, variation, learn_masks, run = config_vals
            group_indexed = group.set_index("Mask")

            if "backbone" not in group_indexed.index:
                continue

            backbone_row = group_indexed.loc["backbone"]

            for dataset in datasets:
                for attr in self.categories:
                    if attr not in group_indexed.index:
                        continue

                    col_name = f"{dataset}_{attr}"
                    if col_name not in group_indexed.columns:
                        continue

                    mask_row = group_indexed.loc[attr]
                    own_effect = float(mask_row[col_name] - backbone_row[col_name])

                    cross_effects = [
                        float(mask_row[f"{dataset}_{a}"] - backbone_row[f"{dataset}_{a}"])
                        for a in self.categories
                        if a != attr and f"{dataset}_{a}" in group_indexed.columns
                    ]

                    if not cross_effects:
                        continue

                    mean_cross = sum(cross_effects) / len(cross_effects)
                    specificity = own_effect - mean_cross

                    spec_records.append(
                        {
                            "Dataset": dataset.capitalize(),
                            "Backbone": backbone,
                            "Pooling": pooling,
                            "Variation": variation,
                            "Learn Masks": learn_masks,
                            "Run": run,
                            "Attribute": attr,
                            "specificity": specificity,
                        }
                    )

        df_spec_runs = pd.DataFrame(spec_records)
        if df_spec_runs.empty:
            return pd.DataFrame()

        df_overall_runs = (
            df_spec_runs.groupby(["Dataset", "Backbone", "Pooling", "Variation", "Learn Masks", "Run"])["specificity"]
            .mean()
            .reset_index()
        )

        df_agg = (
            df_overall_runs.groupby(["Backbone", "Pooling", "Variation", "Learn Masks", "Dataset"])["specificity"]
            .agg(["mean", "std"])
            .reset_index()
        )

        df_wide_summary = df_agg.pivot(
            index=["Backbone", "Pooling", "Variation", "Learn Masks"],
            columns="Dataset",
            values=["mean", "std"],
        )

        df_wide_summary.columns = [
            f"{ds.lower()}_specificity_{stat}"
            for stat, ds in df_wide_summary.columns
        ]
        df_wide_summary = df_wide_summary.reset_index()
        df_wide_summary.columns.name = None

        return df_wide_summary

    def per_mask_triplet_performance(self):
        """Generates performance breakdown for individual masks under fixed MLP mean-pooling settings.

        What it does:
            Filters wide results for fixed_mask + mean pooling + MLP variation,
            excluding random masks. Averages raw triplet metric scores across the 5 runs
            per Backbone, Variation, and Mask.

        Output:
            pd.DataFrame: Mean performance scores per Backbone, Variation, and Mask
            for each dataset metric (e.g., jamendo_overall, jamendo_instrument, mtt_genre).
        """
        filter_mask = (
            (self.df_all_wide["Learn Masks"] == "fixed_mask")
            & (self.df_all_wide["Pooling"] == "mean")
            & (self.df_all_wide["Variation"] == "mlp")
            & (self.df_all_wide["Mask"] != "random")
        )
        df_filtered = self.df_all_wide[filter_mask]

        id_cols = ["Backbone", "Pooling", "Variation", "Learn Masks", "Run", "Mask"]
        metric_cols = [col for col in df_filtered.columns if col not in id_cols]

        return (
            df_filtered.groupby(["Backbone", "Variation", "Mask"])[metric_cols]
            .mean()
            .reset_index()
        )

    def trained_triplet_performance(self):
        """Computes diagonal attribute performance for trained/masked model variants.

        What it does:
            Filters rows where Mask matches attribute target (instrument, genre, mood).
            Extracts diagonal scores (e.g., instrument mask score on instrument metric),
            averages across categories per run, and aggregates mean and std across 5 runs.

        Output:
            pd.DataFrame: Columns for Backbone, Pooling, Variation, Learn Masks,
            jamendo_macro_avg_mean, jamendo_macro_avg_std, mtt_macro_avg_mean, and mtt_macro_avg_std.
        """
        df_filtered = self.df_all_wide[self.df_all_wide["Mask"].isin(self.categories)].copy()

        conds = [df_filtered["Mask"] == cat for cat in self.categories]
        df_filtered["jamendo_score"] = np.select(conds, [df_filtered[f"jamendo_{cat}"] for cat in self.categories])
        df_filtered["mtt_score"] = np.select(conds, [df_filtered[f"mtt_{cat}"] for cat in self.categories])

        run_cols = ["Backbone", "Pooling", "Variation", "Learn Masks", "Run"]
        df_run_avg = (
            df_filtered.groupby(run_cols)[["jamendo_score", "mtt_score"]]
            .mean()
            .reset_index()
        )

        model_cols = ["Backbone", "Pooling", "Variation", "Learn Masks"]
        df_final = (
            df_run_avg.groupby(model_cols)[["jamendo_score", "mtt_score"]]
            .agg(["mean", "std"])
            .reset_index()
        )

        df_final.columns = ["Backbone", "Pooling", "Variation", "Learn Masks", "jamendo_macro_avg_mean", 
                            "jamendo_macro_avg_std", "mtt_macro_avg_mean", "mtt_macro_avg_std"]
        
        return df_final

    def baseline_triplet_performance(self):
        """Extracts unmasked backbone performance baselines across datasets.

        What it does:
            Filters for non-finetuned backbone rows (mean pooling, variation='none',
            Mask='backbone'). Computes macro averages across target categories per run
            and aggregates mean ± std across 5 runs.

        Output:
            pd.DataFrame: Columns for Backbone, Pooling, Variation, Learn Masks,
            jamendo_macro_avg_mean, jamendo_macro_avg_std, mtt_macro_avg_mean, and mtt_macro_avg_std.
        """
        filter_mask = (
            (self.df_all_wide["Backbone"].isin(["muq", "clap", "mert"]))
            & (self.df_all_wide["Pooling"] == "mean")
            & (self.df_all_wide["Variation"] == "none")
            & (
                self.df_all_wide["Learn Masks"].isin(
                    ["learned_mask", True, "True"]
                )
            )
            & (self.df_all_wide["Mask"] == "backbone")
        )

        df_filtered = self.df_all_wide[filter_mask].copy()

        jamendo_cols = [f"jamendo_{cat}" for cat in self.categories]
        mtt_cols = [f"mtt_{cat}" for cat in self.categories]

        df_filtered["jamendo_macro_avg"] = df_filtered[jamendo_cols].mean(axis=1)
        df_filtered["mtt_macro_avg"] = df_filtered[mtt_cols].mean(axis=1)

        group_cols = ["Backbone", "Pooling", "Variation", "Learn Masks"]
        df_backbone_unmasked = (
            df_filtered.groupby(group_cols)[["jamendo_macro_avg", "mtt_macro_avg"]]
            .agg(["mean", "std"])
            .reset_index()
        )

        df_backbone_unmasked.columns = ["Backbone", "Pooling", "Variation", "Learn Masks", "jamendo_macro_avg_mean", 
                                        "jamendo_macro_avg_std", "mtt_macro_avg_mean", "mtt_macro_avg_std"]

        return df_backbone_unmasked
    
    def triplet_performance(self):
            """Combines baseline and trained triplet performance summaries into a
            single table.

            What it does:
                Calls `baseline_triplet_performance()` and `trained_triplet_performance()`
                and vertically concatenates their outputs into a unified performance table.

            Output:
                pd.DataFrame: A combined DataFrame containing macro average performance
                (mean and std across 5 runs) for both unmasked baseline backbones and
                trained masked model configurations.
            """
            
            df_baseline = self.baseline_triplet_performance()
            df_trained = self.trained_triplet_performance()

            # Concatenate tables vertically and reset index
            df_combined = pd.concat([df_baseline, df_trained], ignore_index=True)

            return df_combined


if __name__ == "__main__":
    generator = TableGenerator(results_dir="runs-final")

    df_specificity_summary = generator.compute_specificity()
    df_per_mask_summary = generator.per_mask_triplet_performance()
    df_triplet_performance_summary = generator.triplet_performance()

    #save as csv
    df_specificity_summary.to_csv("table_specificity.csv", index=False)
    df_per_mask_summary.to_csv("table_per_mask.csv", index=False)
    df_triplet_performance_summary.to_csv("table_triplet_performance.csv", index=False)