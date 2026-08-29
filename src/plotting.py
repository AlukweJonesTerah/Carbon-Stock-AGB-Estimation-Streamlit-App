import io
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def make_scatter_plot(actual, predicted, title, color="#2d6a4f"):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(actual, predicted, alpha=0.45, s=22, color=color, edgecolors="none")
    max_val = max(max(actual, default=1), max(predicted, default=1)) * 1.08
    ax.plot([0, max_val], [0, max_val], color="#e63946", linewidth=1.2, linestyle="--", label="1:1 line")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("Actual (t C/ha)")
    ax.set_ylabel("Predicted (t C/ha)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


import textwrap

def build_report_pdf(params, validation_results=None, zonal_df=None, map_briefing=None):
    """Create a robust, multi-page downloadable PDF report including AI briefings and visual charts."""
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        # --- PAGE 1: Summary & AI Briefing ---
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "Carbon Stock & AGB Estimation Report", fontsize=18, fontweight="bold", color="#1a472a")
        fig.text(0.08, 0.91, "Analysis Configuration & Executive Summary", fontsize=12, color="#2d6a4f")
        
        summary = [
            f"Reference year: {params['agb_year']}",
            f"Counties: {', '.join(params['county_selection'])}",
            f"Sample pixels: {params['num_pixels']:,}",
            f"Training split: {params['train_split']:.0%}",
            f"Preset: {params.get('preset', 'Custom')}",
        ]
        fig.text(0.08, 0.85, "\n".join(summary), fontsize=10, va="top", linespacing=1.6)
        
        if map_briefing:
            # Clean markdown and wrap text for PDF
            clean_briefing = map_briefing.replace("**", "").replace("✨", "")
            wrapped_briefing = textwrap.fill(clean_briefing, width=90)
            fig.text(0.08, 0.70, "AI-Generated Executive Briefing", fontsize=13, fontweight="bold", color="#1a472a")
            fig.text(0.08, 0.67, wrapped_briefing, fontsize=10, va="top", linespacing=1.4)
            
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        
        # --- PAGE 2: Metrics & Zonal Charts ---
        if validation_results or (zonal_df is not None and not zonal_df.empty):
            fig2 = plt.figure(figsize=(8.27, 11.69))
            fig2.text(0.08, 0.94, "Data Validation & Zonal Statistics", fontsize=18, fontweight="bold", color="#1a472a")
            
            if validation_results:
                metrics_df = pd.DataFrame(
                    {name: {"RMSE": r["rmse"], "MAE": r["mae"], "Bias": r["bias"], "MAPE": r["mape"], "R²": r["r2"]}
                     for name, r in validation_results.items()}
                ).T.apply(pd.to_numeric, errors="coerce")
                fig2.text(0.08, 0.88, "Held-out ML Validation Metrics", fontsize=13, fontweight="bold", color="#1a472a")
                table_ax = fig2.add_axes((0.08, 0.70, 0.84, 0.15))
                table_ax.axis("off")
                # Render table with column headers
                table = table_ax.table(cellText=metrics_df.round(3).values, rowLabels=metrics_df.index, colLabels=metrics_df.columns, loc="center")
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.5)
                
            if zonal_df is not None and not zonal_df.empty:
                fig2.text(0.08, 0.65, "Top 10 Counties by Mean Carbon Density", fontsize=13, fontweight="bold", color="#1a472a")
                
                # Draw a visual bar chart on the PDF
                chart_ax = fig2.add_axes((0.15, 0.35, 0.7, 0.25))
                top_10 = zonal_df.head(10).sort_values("Mean (t/ha)", ascending=True)
                chart_ax.barh(top_10["County"], top_10["Mean (t/ha)"], color="#52b788")
                chart_ax.set_xlabel("Mean Carbon Stock (t C/ha)")
                chart_ax.grid(axis='x', linestyle='--', alpha=0.7)
                
                # Draw Zonal Table below it
                fig2.text(0.08, 0.30, "County Estimates (Raw Data)", fontsize=11, fontweight="bold", color="#1a472a")
                table_ax2 = fig2.add_axes((0.08, 0.05, 0.84, 0.22))
                table_ax2.axis("off")
                report_rows = zonal_df.head(10)[["County", "Mean (t/ha)", "Sum (t)"]].copy()
                table2 = table_ax2.table(cellText=report_rows.round(2).values, colLabels=report_rows.columns, loc="center")
                table2.auto_set_font_size(False)
                table2.set_fontsize(9)
                table2.scale(1, 1.2)
                
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)
            
    return buffer.getvalue()


def build_portfolio_pdf(entries):
    """Create a concise combined report from saved in-session project snapshots."""
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.07, 0.93, "Carbon Project Portfolio", fontsize=20,
                 fontweight="bold", color="#1a472a")
        fig.text(0.07, 0.89, "Comparison of saved analysis snapshots", fontsize=11, color="#2d6a4f")
        table_rows = [
            [
                entry["Project"], entry["Counties"], entry["Reference year"],
                entry["Samples"], entry["Best model"], entry["Best RMSE"],
                entry["Mean spread"],
            ]
            for entry in entries
        ]
        table_ax = fig.add_axes((0.07, 0.22, 0.86, 0.56))
        table_ax.axis("off")
        table_ax.table(
            cellText=table_rows,
            colLabels=["Project", "Counties", "Year", "Samples", "Best model", "Best RMSE", "Mean spread"],
            loc="center", cellLoc="left",
        )
        fig.text(
            0.07, 0.10,
            "Caution: these are model-analysis snapshots. They are not verified carbon-credit, valuation, or certification records.",
            fontsize=9, color="#5a6e63",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    return buffer.getvalue()


def build_map_briefing(params, validation_results=None, mean_spread=None, zonal_df=None):
    """Create a transparent plain-language briefing from already computed results."""
    counties = list(params.get("county_selection", []))
    county_phrase = ", ".join(counties[:4])
    if len(counties) > 4:
        county_phrase += f", and {len(counties) - 4} more"

    strongest_model = None
    strongest_rmse = None
    if validation_results:
        metric_rows = []
        for name, result in validation_results.items():
            try:
                metric_rows.append((name, float(result.get("rmse"))))
            except (TypeError, ValueError):
                continue
        if metric_rows:
            strongest_model, strongest_rmse = min(metric_rows, key=lambda item: item[1])

    briefing = [
        "### Plain-language map briefing",
        f"This run estimates above-ground carbon across **{len(counties)} selected county/counties**: {county_phrase}. "
        f"It uses the **{params.get('agb_year')} AGB reference year** and {params.get('num_pixels', 0):,} sampled pixels.",
    ]
    if strongest_model:
        briefing.append(
            f"**Strongest tested model:** {strongest_model} had the lowest held-out RMSE "
            f"({strongest_rmse:.2f} t C/ha) in this run. This makes it the most accurate of the tested models here, "
            "not a guarantee that every map pixel is correct."
        )
    else:
        briefing.append(
            "**Model strength:** validation has not been computed yet. Open the Validation tab and run its first step "
            "before choosing a preferred model."
        )
    if mean_spread is not None:
        try:
            spread = float(mean_spread)
            confidence_note = "fairly close agreement" if spread < 10 else "moderate disagreement" if spread < 20 else "substantial disagreement"
            briefing.append(
                f"**Uncertainty:** the average model spread is {spread:.2f} t C/ha, indicating {confidence_note} between the models. "
                "Use the confidence layer to see where that agreement changes across the map."
            )
        except (TypeError, ValueError):
            pass
    else:
        briefing.append(
            "**Uncertainty:** model spread has not been calculated yet. Use Model Comparison → Compute mean model spread "
            "for an uncertainty statement."
        )
    if zonal_df is not None and not zonal_df.empty and "Mean (t/ha)" in zonal_df:
        ranked = zonal_df.dropna(subset=["Mean (t/ha)"]).sort_values("Mean (t/ha)")
        if not ranked.empty:
            low = ranked.iloc[0]
            high = ranked.iloc[-1]
            try:
                briefing.append(
                    f"**County hotspots:** {high['County']} has the highest estimated mean carbon stock "
                    f"({float(high['Mean (t/ha)']):.1f} t C/ha), while {low['County']} has the lowest "
                    f"({float(low['Mean (t/ha)']):.1f} t C/ha) for the selected zonal model."
                )
            except (TypeError, ValueError):
                briefing.append("**County hotspots:** county statistics are available, but their numeric values could not be interpreted for this briefing.")
    else:
        briefing.append(
            "**County hotspots:** zonal statistics have not been computed yet. Run them to identify the highest and lowest "
            "county estimates rather than judging by colour alone."
        )
    briefing.append(
        "**Caution:** this is a satellite-and-model estimate, not a field inventory or verified carbon-credit assessment. "
        "Use it to prioritise investigation, then confirm important decisions with local knowledge and field data."
    )
    return "\n\n".join(briefing)

