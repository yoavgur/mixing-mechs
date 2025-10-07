import numpy as np
import pandas as pd
from typing import Literal
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import to_rgba


def plot_patch_effect(
    df: pd.DataFrame,
    *,
    percent_mode: Literal["none", "grouped", "all"] = "grouped",
    highest_near_pos: int = 3,
    lexical_color: str = "#1db377",
    include_lexical: bool = True,
    include_reflexive: bool = False,
    include_no_effect: bool = False,
    include_invalid: bool = False,
    reflexive_color: str = "#fdb462",  # pastel orange
    no_effect_color: str = "#f08080",  # light coral
    key_on_top: bool = True,
    xlabel: str = "Patched Entity Group Index",
    ylabel: str = "Patch Effect",
    pastel_range: tuple = (0.8, 0.4),
    sep_line_width: float = 0.2,
    sep_line_alpha: float = 0.8,
    annotate_fontsize: int = 8,
    min_label: float = 0.05,  # only show labels if proportion > min_label
    lexical_label: str = "lexical",
    reflexive_label: str = "lexical (reflexive)",
    title: str = "Patch Effect",
    ax=None,
    figsize=(10, 6),
    annotate_color: str = "black",
    annotate_alpha: float = 0.85,
    xticks: list[int] = None,
    param_rot: int = 90,
    yticks: list[float] = None,
    patch_effect_key="patch_effect",
    positional_index_key="positional_index",
):
    """
    Fold `mixed` into positional by |distance|. Plot stacked-area over positional_index with:
      - Positional bins: d=0,1,...,highest_near_pos and d>highest_near_pos ("mixed")
      - lexical optionally shown; optionally reflexive/no_effect
    percent_mode:
      - "none": no labels
      - "grouped": label sum(pos_d0..pos_dN), label mixed (pos_d_gt), and label each of lexical/reflexive/no_effect
      - "all": label every visible band (pos_d0, pos_d1, ..., pos_d_gt, lexical, reflexive, no_effect)
    """
    df2 = df.copy()

    # distances
    df2["_pos_dist"] = np.where(df2[patch_effect_key].eq("positional"), 0, np.nan)
    df2.loc[df2[patch_effect_key].eq("mixed"), "_pos_dist"] = df2["distance"].abs()

    # bins/labels
    bins = [-0.5] + [i + 0.5 for i in range(highest_near_pos + 1)] + [np.inf]
    labels = ["pos_d0"] + [f"pos_d{i}" for i in range(1, highest_near_pos + 1)] + ["pos_d_gt"]
    df2["_pos_bin"] = pd.cut(df2["_pos_dist"], bins=bins, labels=labels)

    # which non-pos categories to include
    non_pos_included = []
    if include_lexical:
        non_pos_included.append("lexical")
    if include_reflexive:
        non_pos_included.append("reflexive")
    if include_no_effect:
        non_pos_included.append("no_effect")
    if include_invalid:
        non_pos_included.append("invalid")

    positional_index_order = sorted(df2[positional_index_key].unique())

    # counts
    ct_pos = (
        pd.crosstab(df2[positional_index_key], df2["_pos_bin"])
        .reindex(index=positional_index_order, fill_value=0)
        .reindex(columns=labels, fill_value=0)
    )
    mask_other = df2[patch_effect_key].isin(non_pos_included)
    ct_other = (
        pd.crosstab(df2.loc[mask_other, positional_index_key], df2.loc[mask_other, patch_effect_key])
        .reindex(index=positional_index_order, fill_value=0)
        .reindex(columns=non_pos_included, fill_value=0)
    )

    # combine + normalize
    ct = pd.concat([ct_pos, ct_other], axis=1)
    ct_prop = ct.div(ct.sum(axis=1), axis=0).fillna(0)

    # stack order
    if key_on_top:
        others_order = [c for c in ["reflexive", "no_effect", "invalid"] if c in non_pos_included]
        if "lexical" in non_pos_included:
            others_order += ["lexical"]
        stack_order = labels + others_order
    else:
        others_order = []
        if "lexical" in non_pos_included:
            others_order.append("lexical")
        others_order += [c for c in ["reflexive", "no_effect", "invalid"] if c in non_pos_included]
        stack_order = others_order + labels

    # colors
    blues = get_cmap("Blues")
    blue_cols = [blues(c) for c in np.linspace(pastel_range[0], pastel_range[1], len(labels))]
    color_map = {
        "lexical": lexical_color,
        "reflexive": reflexive_color,
        "no_effect": no_effect_color,
        "invalid": "red",
    }
    colors = [blue_cols[labels.index(cat)] if cat in labels else color_map[cat] for cat in stack_order]

    # legend (align with stack_order)
    def _legend(cat: str) -> str:
        if cat == "pos_d0":
            return "positional (d=0)"
        if cat == "pos_d_gt":
            return "mixed"
        if cat.startswith("pos_d"):
            return f"near-positional (d={cat.split('pos_d')[-1]})"
        if cat == "lexical":
            return lexical_label
        if cat == "reflexive":
            return reflexive_label
        return cat

    legend_labels = [_legend(c) for c in stack_order]

    # plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(figsize[0], figsize[1]))
        created_fig = True
    else:
        fig = ax.figure
        created_fig = False

    ax.stackplot(
        ct_prop.index,
        [ct_prop[c] for c in stack_order],
        # labels=legend_labels,
        colors=colors,
        alpha=0.9,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    else:
        ax.set_xticks(list(range(0, max(ct_prop.index) + 1, 2)))

    if yticks is not None:
        ax.set_yticks(yticks)
    ax.grid(True, which="both", axis="x", linestyle="--", alpha=0.5, color="white")
    # ax.grid(linestyle='--', alpha=0.5, color="white")

    # ax.set_xticks(ct_prop.index)
    # ax.set_xticks(list(range(0, max(ct_prop.index) + 1, 2)))
    # ax.set_xticklabels([str(x) for x in ct_prop.index])

    ax.set_xlim(min(ct_prop.index), max(ct_prop.index))
    ax.set_ylim(0, 1)

    # labeling
    for idx in ct_prop.index:
        if percent_mode == "none":
            continue

        # precompute for grouped
        pos_group_val = float(ct_prop.loc[idx, labels[:-1]].sum())  # d0..dN
        vals_map = {cat: float(ct_prop.loc[idx, cat]) for cat in stack_order}

        # Only put a label if there is an xtick at this idx
        xticks_set = set(ax.get_xticks())
        if percent_mode == "grouped":
            y_offset = 0.0
            # track whether we've placed the group label
            placed_group = False
            for cat in stack_order:
                v = vals_map.get(cat, 0.0)
                if v > 0 and idx in xticks_set:
                    if (cat == "pos_d0") and (pos_group_val > min_label) and not placed_group:
                        ax.text(
                            idx,
                            y_offset + pos_group_val / 2,
                            f"{pos_group_val*100:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=annotate_fontsize,
                            color=annotate_color,
                            alpha=annotate_alpha,
                            rotation=param_rot,
                        )
                        placed_group = True
                    if cat == "pos_d_gt" and v > min_label:
                        ax.text(
                            idx,
                            y_offset + v / 2,
                            f"{v*100:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=annotate_fontsize,
                            color=annotate_color,
                            alpha=annotate_alpha,
                            rotation=param_rot,
                        )
                    if cat == "lexical" and v > min_label:
                        ax.text(
                            idx,
                            y_offset + v / 2,
                            f"{v*100:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=annotate_fontsize,
                            color=annotate_color,
                            alpha=annotate_alpha,
                            rotation=param_rot,
                        )
                    if cat == "reflexive" and v > min_label:
                        ax.text(
                            idx,
                            y_offset + v / 2,
                            f"{v*100:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=annotate_fontsize,
                            color=annotate_color,
                            alpha=annotate_alpha,
                            rotation=param_rot,
                        )
                    if cat == "invalid" and v > min_label:
                        ax.text(
                            idx,
                            y_offset + v / 2,
                            f"{v*100:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=annotate_fontsize,
                            color=annotate_color,
                            alpha=annotate_alpha,
                            rotation=param_rot,
                        )
                    if cat == "no_effect" and v > min_label:
                        ax.text(
                            idx,
                            y_offset + v / 2,
                            f"{v*100:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=annotate_fontsize,
                            color=annotate_color,
                            alpha=annotate_alpha,
                            rotation=param_rot,
                        )
                y_offset += v

        elif percent_mode == "all":
            y_offset = 0.0
            for cat in stack_order:
                v = vals_map.get(cat, 0.0)
                if v > min_label and idx in xticks_set:
                    ax.text(
                        idx,
                        y_offset + v / 2,
                        f"{v*100:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=annotate_fontsize,
                        color=annotate_color,
                        alpha=annotate_alpha,
                        rotation=param_rot,
                    )
                y_offset += v

    # separators
    y_accum = np.zeros(len(ct_prop.index))
    for cat in stack_order[:-1]:
        y_accum += ct_prop[cat].values
        ax.plot(
            ct_prop.index, y_accum, color="black", linewidth=sep_line_width, alpha=sep_line_alpha, label="_nolegend_"
        )

    # ax.legend(title="patch effect", ncol=2)
    ax.legend().set_visible(False)
    if created_fig:
        fig.suptitle(title)
        plt.tight_layout()
    return fig, ax
