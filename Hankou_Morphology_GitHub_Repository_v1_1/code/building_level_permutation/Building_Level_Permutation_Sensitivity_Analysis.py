from __future__ import annotations

from pathlib import Path
from collections import Counter
import csv
import math

import numpy as np
from scipy.stats import rankdata, chi2_contingency

BASE = Path(__file__).resolve().parent
OPEN_CSV = BASE / "Building_Level_OpenCV_59_Analysis_Input.csv"
COMP_CSV = BASE / "Building_Level_Component_Proportions_59_Final.csv"

N_PERMUTATIONS = 9999
BASE_SEED = 20260903
CONCESSION_ORDER = ["British", "French", "German", "Russian", "Japanese"]

OPEN_META = {
    "building_id", "building_name_cn", "analysis_concession_cn",
    "analysis_concession_en", "boundary_attribution",
    "source_facade_concessions_cn", "boundary_facade_ids",
    "facade_count", "facade_ids", "address", "year",
}

GROUP_DEFS = {
    "Window": {
        "total": "window_total",
        "count_cols": [
            "window_cluster_1_n",
            "window_cluster_2_n",
            "window_cluster_3_n",
        ],
        "prop_cols": [
            "window_cluster_1_prop",
            "window_cluster_2_prop",
            "window_cluster_3_prop",
        ],
        "labels": [
            "Arched/ornamented tendency",
            "Horizontal-organization tendency",
            "Rectilinear tendency",
        ],
    },
    "Door": {
        "total": "door_total",
        "count_cols": [
            "door_cluster_2_n",
            "door_cluster_1_n",
        ],
        "prop_cols": [
            "door_cluster_2_prop",
            "door_cluster_1_prop",
        ],
        "labels": [
            "Conventional door-leaf grouping",
            "Shutter/grille-feature grouping",
        ],
    },
    "Column": {
        "total": "column_total",
        "count_cols": [
            "column_cluster_1_n",
            "column_cluster_2_n",
        ],
        "prop_cols": [
            "column_cluster_1_prop",
            "column_cluster_2_prop",
        ],
        "labels": [
            "Brick-textured/articulated grouping",
            "Plain-rectilinear grouping",
        ],
    },
}

SCENARIOS = [
    {
        "scenario_id": "S1",
        "scenario": "All five concessions — primary",
        "include_groups": set(CONCESSION_ORDER),
        "exclude_sensitive": False,
        "seed_offset": 100,
    },
    {
        "scenario_id": "S2",
        "scenario": "All five concessions — exclude attribution-sensitive buildings",
        "include_groups": set(CONCESSION_ORDER),
        "exclude_sensitive": True,
        "seed_offset": 200,
    },
    {
        "scenario_id": "S3",
        "scenario": "British–French–Russian only",
        "include_groups": {"British", "French", "Russian"},
        "exclude_sensitive": False,
        "seed_offset": 300,
    },
    {
        "scenario_id": "S4",
        "scenario": "British–French–Russian only — exclude attribution-sensitive buildings",
        "include_groups": {"British", "French", "Russian"},
        "exclude_sensitive": True,
        "seed_offset": 400,
    },
]


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def write_csv(path: Path, headers: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def bh_fdr(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, ranked[i] * n / (i + 1))
        adjusted[order[i]] = min(running, 1.0)
    return adjusted


def make_permuted_labels(labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.permutation(labels) for _ in range(N_PERMUTATIONS)],
        axis=0,
    )


def permutation_kruskal_wallis(
    values: np.ndarray,
    labels: np.ndarray,
    permuted_labels: np.ndarray,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    ranks = rankdata(values, method="average")
    n = len(values)
    groups = np.unique(labels)
    group_sizes = np.array([(labels == g).sum() for g in groups], dtype=float)

    observed_rank_sums = np.array(
        [ranks[labels == g].sum() for g in groups], dtype=float
    )
    base = (
        12.0 / (n * (n + 1.0))
        * np.sum(observed_rank_sums**2 / group_sizes)
        - 3.0 * (n + 1.0)
    )
    _, tie_counts = np.unique(values, return_counts=True)
    tie_correction = 1.0 - np.sum(tie_counts**3 - tie_counts) / (n**3 - n)
    observed_h = base / tie_correction if tie_correction > 0 else 0.0

    perm_accumulator = np.zeros(N_PERMUTATIONS, dtype=float)
    for group, group_size in zip(groups, group_sizes):
        perm_rank_sums = (permuted_labels == group) @ ranks
        perm_accumulator += perm_rank_sums**2 / group_size
    perm_h = (
        12.0 / (n * (n + 1.0)) * perm_accumulator
        - 3.0 * (n + 1.0)
    ) / tie_correction

    p_value = (
        1 + np.sum(perm_h >= observed_h - 1e-12)
    ) / (N_PERMUTATIONS + 1)
    k = len(groups)
    epsilon_squared = max(0.0, (observed_h - k + 1) / (n - k))
    return float(observed_h), float(p_value), float(epsilon_squared)


def permanova_one_factor(
    x: np.ndarray,
    labels: np.ndarray,
    permuted_labels: np.ndarray,
) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels, dtype=int)
    groups = np.unique(labels)
    n = len(x)
    k = len(groups)
    grand_mean = x.mean(axis=0)
    ss_total = np.sum((x - grand_mean) ** 2)

    group_sizes = np.array([(labels == g).sum() for g in groups], dtype=float)
    ss_between = 0.0
    for group, group_size in zip(groups, group_sizes):
        group_mean = x[labels == group].mean(axis=0)
        ss_between += group_size * np.sum((group_mean - grand_mean) ** 2)

    ss_within = ss_total - ss_between
    pseudo_f = (ss_between / (k - 1)) / (ss_within / (n - k))
    r_squared = ss_between / ss_total if ss_total > 0 else 0.0

    perm_ss_between = np.zeros(N_PERMUTATIONS, dtype=float)
    for group, group_size in zip(groups, group_sizes):
        perm_group_sums = (permuted_labels == group) @ x
        perm_group_means = perm_group_sums / group_size
        perm_ss_between += group_size * np.sum(
            (perm_group_means - grand_mean) ** 2, axis=1
        )
    perm_ss_within = ss_total - perm_ss_between
    perm_f = (perm_ss_between / (k - 1)) / (perm_ss_within / (n - k))
    p_value = (
        1 + np.sum(perm_f >= pseudo_f - 1e-12)
    ) / (N_PERMUTATIONS + 1)
    return float(pseudo_f), float(r_squared), float(p_value)


def cluster_permutation_chi_square(
    building_counts: np.ndarray,
    labels: np.ndarray,
    permuted_labels: np.ndarray,
) -> tuple[float, int, float, float, float]:
    counts = np.asarray(building_counts, dtype=float)
    labels = np.asarray(labels, dtype=int)
    groups = np.unique(labels)

    observed_table = np.vstack(
        [counts[labels == group].sum(axis=0) for group in groups]
    )
    keep = observed_table.sum(axis=0) > 0
    counts = counts[:, keep]
    observed_table = observed_table[:, keep]

    chi2, naive_p, df, _ = chi2_contingency(
        observed_table, correction=False
    )
    total_n = observed_table.sum()
    n_rows, n_cols = observed_table.shape
    cramers_v = math.sqrt(
        chi2 / (total_n * min(n_rows - 1, n_cols - 1))
    )

    perm_tables = np.stack(
        [(permuted_labels == group) @ counts for group in groups],
        axis=1,
    )
    row_totals = perm_tables.sum(axis=2, keepdims=True)
    col_totals = perm_tables.sum(axis=1, keepdims=True)
    expected_perm = row_totals * col_totals / total_n
    perm_chi2 = np.sum(
        np.where(
            expected_perm > 0,
            (perm_tables - expected_perm) ** 2 / expected_perm,
            0.0,
        ),
        axis=(1, 2),
    )
    clustered_p = (
        1 + np.sum(perm_chi2 >= chi2 - 1e-12)
    ) / (N_PERMUTATIONS + 1)

    return (
        float(chi2),
        int(df),
        float(naive_p),
        float(cramers_v),
        float(clustered_p),
    )


def scenario_filter(row: dict, scenario: dict) -> bool:
    return (
        row["analysis_concession_en"] in scenario["include_groups"]
        and (
            not scenario["exclude_sensitive"]
            or row["boundary_attribution"] != "Attribution-sensitive"
        )
    )


def ordered_groups(rows: list[dict]) -> list[str]:
    present = {row["analysis_concession_en"] for row in rows}
    return [g for g in CONCESSION_ORDER if g in present]


def main() -> None:
    open_headers, open_rows = read_csv(OPEN_CSV)
    _, comp_rows = read_csv(COMP_CSV)
    open_features = [h for h in open_headers if h not in OPEN_META]

    open_results = []
    permanova_results = []
    proportion_results = []
    chi_results = []
    scenario_results = []

    for scenario_index, scenario in enumerate(SCENARIOS, start=1):
        open_subset = [r for r in open_rows if scenario_filter(r, scenario)]
        comp_subset = [r for r in comp_rows if scenario_filter(r, scenario)]

        groups = ordered_groups(open_subset)
        code = {g: i for i, g in enumerate(groups)}
        labels = np.array(
            [code[r["analysis_concession_en"]] for r in open_subset],
            dtype=int,
        )
        permuted = make_permuted_labels(
            labels, BASE_SEED + scenario["seed_offset"]
        )

        counts = Counter(r["analysis_concession_en"] for r in open_subset)
        scenario_results.append({
            "scenario_id": scenario["scenario_id"],
            "scenario": scenario["scenario"],
            "n_buildings": len(open_subset),
            "British_n": counts.get("British", 0),
            "French_n": counts.get("French", 0),
            "German_n": counts.get("German", 0),
            "Russian_n": counts.get("Russian", 0),
            "Japanese_n": counts.get("Japanese", 0),
        })

        local_open = []
        for feature in open_features:
            x = np.array([float(r[feature]) for r in open_subset], dtype=float)
            h, p, eps = permutation_kruskal_wallis(x, labels, permuted)
            record = {
                "scenario_id": scenario["scenario_id"],
                "scenario": scenario["scenario"],
                "feature": feature,
                "n_buildings": len(open_subset),
                "H": h,
                "epsilon_squared": eps,
                "permutation_p": p,
            }
            open_results.append(record)
            local_open.append(record)
        for record, q in zip(
            local_open,
            bh_fdr([r["permutation_p"] for r in local_open]),
        ):
            record["BH_FDR_p"] = float(q)

        local_perm = []
        local_prop = []
        local_chi = []

        for type_index, (component_type, definition) in enumerate(
            GROUP_DEFS.items(), start=1
        ):
            valid_rows = [
                r for r in comp_subset
                if float(r[definition["total"]]) > 0
            ]
            groups_component = ordered_groups(valid_rows)
            comp_code = {g: i for i, g in enumerate(groups_component)}
            comp_labels = np.array(
                [comp_code[r["analysis_concession_en"]] for r in valid_rows],
                dtype=int,
            )
            comp_permuted = make_permuted_labels(
                comp_labels,
                BASE_SEED + 1000 * scenario_index + 10 * type_index,
            )

            counts_matrix = np.array(
                [
                    [float(r[c]) for c in definition["count_cols"]]
                    for r in valid_rows
                ],
                dtype=float,
            )
            proportions = np.array(
                [
                    [float(r[c]) for c in definition["prop_cols"]]
                    for r in valid_rows
                ],
                dtype=float,
            )
            hellinger = np.sqrt(proportions)

            pseudo_f, r2, p = permanova_one_factor(
                hellinger, comp_labels, comp_permuted
            )
            record = {
                "scenario_id": scenario["scenario_id"],
                "scenario": scenario["scenario"],
                "component_type": component_type,
                "n_buildings": len(valid_rows),
                "pseudo_F": pseudo_f,
                "R_squared": r2,
                "permutation_p": p,
            }
            permanova_results.append(record)
            local_perm.append(record)

            chi2, df, naive_p, v, clustered_p = cluster_permutation_chi_square(
                counts_matrix, comp_labels, comp_permuted
            )
            chi_record = {
                "scenario_id": scenario["scenario_id"],
                "scenario": scenario["scenario"],
                "component_type": component_type,
                "n_buildings": len(valid_rows),
                "total_components": int(counts_matrix.sum()),
                "chi2": chi2,
                "df": df,
                "naive_p": naive_p,
                "cramers_v": v,
                "building_cluster_permutation_p": clustered_p,
            }
            chi_results.append(chi_record)
            local_chi.append(chi_record)

            for label, prop_column in zip(
                definition["labels"], definition["prop_cols"]
            ):
                x = np.array(
                    [float(r[prop_column]) for r in valid_rows],
                    dtype=float,
                )
                h, p, eps = permutation_kruskal_wallis(
                    x, comp_labels, comp_permuted
                )
                prop_record = {
                    "scenario_id": scenario["scenario_id"],
                    "scenario": scenario["scenario"],
                    "component_type": component_type,
                    "visual_grouping": label,
                    "n_buildings": len(valid_rows),
                    "H": h,
                    "epsilon_squared": eps,
                    "permutation_p": p,
                }
                proportion_results.append(prop_record)
                local_prop.append(prop_record)

        for record, q in zip(
            local_perm,
            bh_fdr([r["permutation_p"] for r in local_perm]),
        ):
            record["BH_FDR_p"] = float(q)

        for record, q in zip(
            local_prop,
            bh_fdr([r["permutation_p"] for r in local_prop]),
        ):
            record["BH_FDR_p"] = float(q)

        for record, q in zip(
            local_chi,
            bh_fdr(
                [r["building_cluster_permutation_p"] for r in local_chi]
            ),
        ):
            record["BH_FDR_cluster_p"] = float(q)

    write_csv(
        BASE / "scenario_definitions_results.csv",
        list(scenario_results[0].keys()),
        scenario_results,
    )
    write_csv(
        BASE / "opencv_permutation_kw_results.csv",
        list(open_results[0].keys()),
        open_results,
    )
    write_csv(
        BASE / "component_permanova_results.csv",
        list(permanova_results[0].keys()),
        permanova_results,
    )
    write_csv(
        BASE / "component_proportion_kw_results.csv",
        list(proportion_results[0].keys()),
        proportion_results,
    )
    write_csv(
        BASE / "cluster_aware_chi_square_results.csv",
        list(chi_results[0].keys()),
        chi_results,
    )

    print("Analysis completed.")
    print("Permutations:", N_PERMUTATIONS)
    print("Base seed:", BASE_SEED)
    print("Output CSV files written to:", BASE)


if __name__ == "__main__":
    main()
