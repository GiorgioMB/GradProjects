#!/usr/bin/env python3
import argparse
import os
import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
import base64
from io import BytesIO
from datetime import datetime

try:
    import contextily as cx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False

import config

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#FAFBFC',
    'axes.edgecolor':   '#DEE2E6',
    'axes.grid':        True,
    'grid.color':       '#E9ECEF',
    'grid.linewidth':   0.6,
    'grid.alpha':       0.8,
    'font.family':      'sans-serif',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'axes.labelsize':   11,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
})
sns.set_theme(style="whitegrid", font_scale=1.05)

BRAND = {
    'primary':   '#0B3D91',
    'secondary': '#1B6CB0',
    'accent':    '#00B4D8',
    'success':   '#2DC653',
    'warning':   '#F77F00',
    'danger':    '#D62828',
    'muted':     '#8D99AE',
    'text':      '#2B2D42',
}

TARGET_COLORS = {
    'Total Alkalinity':               '#0B3D91',
    'Electrical Conductance':         '#F77F00',
    'Dissolved Reactive Phosphorus':  '#2DC653',
}
TARGET_CMAPS = {
    'Total Alkalinity':               'Blues',
    'Electrical Conductance':         'Oranges',
    'Dissolved Reactive Phosphorus':  'Greens',
}

PALETTE = sns.color_palette([
    BRAND['primary'], BRAND['warning'], BRAND['success'],
    BRAND['accent'], BRAND['danger'], BRAND['secondary'],
    BRAND['muted'], '#6A4C93', '#1982C4', '#FFCA3A',
], 10)


# UTILITIES
def fig_to_base64(fig, dpi=140):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _get_feature_cols(df, targets):
    exclude = set(targets) | {
        'Latitude', 'Longitude', 'Sample Date', 'date',
        '_dt', '_loc_id', '_geo_key', 'key', '_lat_r', '_lon_r',
        '_dws_station', '_merge_date', 'station',
    }
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in exclude]


def _safe_impute(X):
    imp = SimpleImputer(strategy='median')
    return imp.fit_transform(X)


def _safe_section(func, *args, fallback_title="Section", **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        traceback.print_exc()
        return (f'<h2>{fallback_title}</h2>'
                f'<p class="warn-box">⚠ Section failed: <code>{e}</code></p>')


# SECTION GENERATORS  (each returns an HTML string)
def section_overview(df, targets, features):
    n_rows, n_cols = df.shape
    n_features = len(features)
    n_missing = df[features].isna().sum().sum()
    pct_missing = 100 * n_missing / (n_rows * n_features) if n_features else 0
    has_geo = 'Latitude' in df.columns and 'Longitude' in df.columns
    n_locations = 0
    if has_geo:
        n_locations = df.groupby(
            [df['Latitude'].round(3), df['Longitude'].round(3)]
        ).ngroups

    # Missing-value lollipop chart
    show_feats = features[:60]
    missing_pct = df[show_feats].isna().mean().sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    fig, ax = plt.subplots(figsize=(min(18, len(missing_pct) * 0.4 + 3), 5))
    if len(missing_pct) > 0:
        x = np.arange(len(missing_pct))
        colours = [BRAND['danger'] if v > 0.5 else
                   BRAND['warning'] if v > 0.2 else BRAND['accent']
                   for v in missing_pct.values]
        ax.vlines(x, 0, missing_pct.values, colors=colours, lw=2.5)
        ax.scatter(x, missing_pct.values, color=colours, s=40, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(missing_pct.index, rotation=70, ha='right', fontsize=8)
        ax.set_ylabel('Fraction Missing')
        ax.set_title('Missing-Value Profile (features with any NaN)')
        ax.axhline(0.5, color=BRAND['danger'], ls='--', alpha=0.4, label='50 %')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No missing values - dataset is complete!',
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        ax.set_title('Missing Values')
    img = fig_to_base64(fig)

    target_stats = df[targets].describe().round(2).to_html(
        classes='styled-table', border=0)

    html = f"""
    <h2>1 · Executive Summary</h2>
    <div class="stats-grid">
      <div class="stat-card"><span class="stat-num">{n_rows:,}</span><br>Rows</div>
      <div class="stat-card"><span class="stat-num">{n_cols}</span><br>Total Columns</div>
      <div class="stat-card"><span class="stat-num">{n_features}</span><br>Numeric Features</div>
      <div class="stat-card"><span class="stat-num">{len(targets)}</span><br>Targets</div>
      <div class="stat-card"><span class="stat-num">{pct_missing:.1f}%</span><br>Missing Values</div>
      <div class="stat-card"><span class="stat-num">{n_locations}</span><br>Unique Locations</div>
    </div>
    <h3>Target Summary Statistics</h3>
    {target_stats}
    <h3>Missing-Value Profile</h3>
    <img src="data:image/png;base64,{img}" style="max-width:100%">
    """
    return html


def section_target_distributions(df, targets):
    # Violin + strip (raincloud-style)
    fig, axes = plt.subplots(2, len(targets), figsize=(6.5 * len(targets), 11))
    if len(targets) == 1:
        axes = axes.reshape(-1, 1)

    for i, t in enumerate(targets):
        vals = df[t].dropna()
        color = TARGET_COLORS.get(t, PALETTE[i])

        # Top row: violin + strip
        parts = axes[0, i].violinplot([vals.values], positions=[0],
                                       showmedians=True, widths=0.8)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.35)
        for k in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if k in parts:
                parts[k].set_color(color)
        jitter = np.random.default_rng(42).normal(0, 0.06, len(vals))
        axes[0, i].scatter(jitter, vals.values, alpha=0.08, s=4,
                           color=color, zorder=4)
        sk = vals.skew()
        axes[0, i].set_title(f'{t}  (n={len(vals):,})')
        axes[0, i].set_ylabel('Value')
        axes[0, i].set_xticks([])
        axes[0, i].annotate(
            f'skew {sk:+.2f}  · med {vals.median():.1f}',
            xy=(0.02, 0.97), xycoords='axes fraction', va='top',
            fontsize=9, bbox=dict(boxstyle='round', fc='white', alpha=0.85))

        # Bottom row: QQ plot
        (osm, osr), (slope, intercept, _) = stats.probplot(vals, dist='norm')
        axes[1, i].scatter(osm, osr, s=6, alpha=0.4, color=color)
        x_line = np.array([osm.min(), osm.max()])
        axes[1, i].plot(x_line, slope * x_line + intercept,
                        color=BRAND['danger'], lw=1.5, ls='--', label='Normal ref.')
        axes[1, i].set_title(f'Q-Q Plot - {t}')
        axes[1, i].set_xlabel('Theoretical Quantiles')
        axes[1, i].set_ylabel('Sample Quantiles')
        axes[1, i].legend(fontsize=8)

    fig.suptitle('Target Deep-Dive - Distribution & Normality', fontsize=14, y=1.01)
    fig.tight_layout()
    img1 = fig_to_base64(fig)

    # Hex-scatter pairwise
    n_tgt = len(targets)
    n_pairs = n_tgt * (n_tgt - 1) // 2
    fig2, axes2 = plt.subplots(1, max(n_pairs, 1),
                                figsize=(6.5 * max(n_pairs, 1), 5.5))
    if n_pairs <= 1:
        axes2 = [axes2]
    idx = 0
    for i in range(n_tgt):
        for j in range(i + 1, n_tgt):
            valid = df[[targets[i], targets[j]]].dropna()
            ax = axes2[idx]
            hb = ax.hexbin(valid[targets[i]], valid[targets[j]],
                           gridsize=35, cmap='YlGnBu', mincnt=1, linewidths=0.2)
            plt.colorbar(hb, ax=ax, label='Count', shrink=0.8)
            r, _ = stats.spearmanr(valid[targets[i]], valid[targets[j]])
            ax.set_xlabel(targets[i])
            ax.set_ylabel(targets[j])
            ax.set_title(f'Spearman ρ = {r:.3f}')
            idx += 1
    fig2.suptitle('Target Pairwise Relationships (hex-density)',
                  fontsize=14, y=1.02)
    fig2.tight_layout()
    img2 = fig_to_base64(fig2)

    return f"""
    <h2>2 · Target Deep-Dive</h2>
    <p>Violin + strip plots reveal shape &amp; outliers; Q-Q plots test normality.
    Hex-scatter avoids over-plotting.</p>
    <img src="data:image/png;base64,{img1}" style="max-width:100%">
    <h3>Pairwise Hex-Density</h3>
    <img src="data:image/png;base64,{img2}" style="max-width:100%">
    """


def section_correlations(df, targets, features, top_k=25):
    corr_data = {}
    for t in targets:
        corrs = []
        for f in features:
            valid = df[[f, t]].dropna()
            if len(valid) < 10:
                corrs.append(0)
            else:
                r, _ = stats.spearmanr(valid[f], valid[t])
                corrs.append(r if np.isfinite(r) else 0)
        corr_data[t] = corrs
    corr_df = pd.DataFrame(corr_data, index=features)

    # Lollipop charts per target
    images = []
    for t in targets:
        top_feats = corr_df[t].abs().nlargest(top_k).index
        vals = corr_df.loc[top_feats, t]
        fig, ax = plt.subplots(figsize=(9, max(6, top_k * 0.32)))
        y_pos = np.arange(len(vals))
        colours = [TARGET_COLORS.get(t, BRAND['primary']) if v > 0
                   else BRAND['danger'] for v in vals]
        ax.hlines(y_pos, 0, vals.values, colors=colours, lw=2.5)
        ax.scatter(vals.values, y_pos, color=colours, s=50, zorder=5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(vals.index, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Spearman ρ')
        ax.set_title(f'Top-{top_k} Features → {t}')
        ax.axvline(0, color='black', lw=0.5)
        fig.tight_layout()
        images.append(fig_to_base64(fig))

    # Clustered heatmap (top union)
    all_top = set()
    for t in targets:
        all_top.update(corr_df[t].abs().nlargest(top_k).index)
    all_top = sorted(all_top)[:50]
    fig3, ax3 = plt.subplots(figsize=(8, max(8, len(all_top) * 0.3)))
    sns.heatmap(corr_df.loc[all_top], annot=False, cmap='RdBu_r', center=0,
                vmin=-0.7, vmax=0.7, ax=ax3, linewidths=0.3)
    ax3.set_title('Feature–Target Correlation Heatmap (top features)')
    fig3.tight_layout()
    img_heatmap = fig_to_base64(fig3)

    img_html = ''.join(
        f'<img src="data:image/png;base64,{i}" style="max-width:100%;margin-bottom:15px;">'
        for i in images)
    return f"""
    <h2>3 · Feature–Target Correlations (Spearman)</h2>
    <p>Lollipop charts highlight the strongest monotonic predictors.
    <span style="color:{BRAND['primary']}">■</span> positive &nbsp;
    <span style="color:{BRAND['danger']}">■</span> negative</p>
    {img_html}
    <h3>Cross-Target Heatmap</h3>
    <img src="data:image/png;base64,{img_heatmap}" style="max-width:100%">
    """


def section_mutual_info(df, targets, features, top_k=25):
    X = df[features].copy()
    X_imp = _safe_impute(X)

    mi_results = {}
    images = []
    for t in targets:
        y = df[t].copy()
        mask = y.notna()
        mi = mutual_info_regression(X_imp[mask], y[mask], random_state=42,
                                     n_neighbors=5)
        mi_series = pd.Series(mi, index=features).nlargest(top_k)
        mi_results[t] = mi_series

        fig, ax = plt.subplots(figsize=(9, max(6, top_k * 0.32)))
        y_pos = np.arange(len(mi_series))
        ax.barh(y_pos, mi_series.values,
                color=TARGET_COLORS.get(t, PALETTE[0]),
                edgecolor='white', height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(mi_series.index, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Mutual Information (nats)')
        ax.set_title(f'Top-{top_k} MI → {t}')
        fig.tight_layout()
        images.append(fig_to_base64(fig))

    # Radar overlay: top-8 features shared across targets
    union_top = set()
    for t in targets:
        union_top.update(mi_results[t].nlargest(8).index)
    union_top = sorted(union_top)[:10]
    if len(union_top) >= 3:
        angles = np.linspace(0, 2 * np.pi, len(union_top), endpoint=False).tolist()
        angles += angles[:1]
        fig_r, ax_r = plt.subplots(figsize=(7, 7), subplot_kw={'polar': True})
        for t in targets:
            vals = [mi_results[t].get(f, 0) for f in union_top]
            mx = max(vals) if max(vals) > 0 else 1
            vals_norm = [v / mx for v in vals]
            vals_norm += vals_norm[:1]
            ax_r.plot(angles, vals_norm, lw=2,
                      label=t, color=TARGET_COLORS.get(t, PALETTE[0]))
            ax_r.fill(angles, vals_norm, alpha=0.10,
                      color=TARGET_COLORS.get(t, PALETTE[0]))
        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(union_top, fontsize=8)
        ax_r.set_title('MI Radar - Normalised Feature Importance', pad=20)
        ax_r.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        fig_r.tight_layout()
        img_radar = fig_to_base64(fig_r)
        radar_html = (f'<h3>MI Radar Overlay</h3>'
                      f'<img src="data:image/png;base64,{img_radar}" '
                      f'style="max-width:700px;">')
    else:
        radar_html = ''

    img_html = ''.join(
        f'<img src="data:image/png;base64,{i}" style="max-width:100%;margin-bottom:15px;">'
        for i in images)
    return f"""
    <h2>4 · Mutual Information (Nonlinear Predictive Power)</h2>
    <p>MI captures <em>any</em> statistical dependency - not just monotonic.
    Higher ≈ more informative.  Zero ≈ independent.</p>
    {img_html}
    {radar_html}
    """

def section_feature_distributions(df, targets, features, top_k=12):
    avg_corr = pd.Series(0.0, index=features)
    for t in targets:
        for f in features:
            valid = df[[f, t]].dropna()
            if len(valid) >= 10:
                r, _ = stats.spearmanr(valid[f], valid[t])
                avg_corr[f] += abs(r) if np.isfinite(r) else 0
    avg_corr /= max(len(targets), 1)
    top_feats = avg_corr.nlargest(top_k).index.tolist()

    n_c = 4
    n_r = (len(top_feats) + n_c - 1) // n_c
    fig, axes = plt.subplots(n_r, n_c, figsize=(4.2 * n_c, 3.5 * n_r))
    axes_flat = axes.flatten()
    for i, feat in enumerate(top_feats):
        vals = df[feat].dropna()
        axes_flat[i].hist(vals, bins=40, color=PALETTE[4], alpha=0.8,
                          edgecolor='white')
        axes_flat[i].set_title(feat, fontsize=10)
        axes_flat[i].tick_params(labelsize=8)
    for j in range(len(top_feats), len(axes_flat)):
        axes_flat[j].axis('off')
    fig.suptitle(f'Top-{top_k} Most Predictive Feature Distributions', fontsize=13)
    fig.tight_layout()
    img_dist = fig_to_base64(fig)

    X_imp = _safe_impute(df[features])
    pca = PCA(n_components=2, random_state=42)
    pc = pca.fit_transform(X_imp)
    t0 = targets[0]
    vals = df[t0].values
    mask = np.isfinite(vals)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sc = ax2.scatter(pc[mask, 0], pc[mask, 1], c=vals[mask],
                     cmap=TARGET_CMAPS.get(t0, 'viridis'), s=6, alpha=0.5)
    plt.colorbar(sc, ax=ax2, label=t0, shrink=0.8)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax2.set_title('PCA 2-D Projection (colour = first target)')
    fig2.tight_layout()
    img_pca = fig_to_base64(fig2)

    return f"""
    <h2>5 · Feature Distributions &amp; PCA</h2>
    <p>Top {top_k} features by avg |Spearman ρ|, plus a 2-D PCA embedding.</p>
    <img src="data:image/png;base64,{img_dist}" style="max-width:100%">
    <h3>PCA 2-D Embedding</h3>
    <img src="data:image/png;base64,{img_pca}" style="max-width:100%">
    """

def _add_basemap(ax, zoom='auto'):
    if not HAS_CONTEXTILY:
        return
    try:
        cx.add_basemap(ax, crs='EPSG:4326', zoom=zoom,
                       source=cx.providers.Esri.WorldImagery,
                       attribution='', alpha=0.55)
    except Exception:
        pass


def section_spatial(df, targets):
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        return "<h2>6 · Spatial Analysis</h2><p>No coordinates found.</p>"

    images = []
    for t in targets:
        fig, ax = plt.subplots(figsize=(12, 9))
        valid = df[['Latitude', 'Longitude', t]].dropna()
        hb = ax.hexbin(valid['Longitude'], valid['Latitude'],
                       C=valid[t], gridsize=30,
                       cmap=TARGET_CMAPS.get(t, 'viridis'),
                       reduce_C_function=np.median,
                       mincnt=1, linewidths=0.2, zorder=5)
        plt.colorbar(hb, ax=ax, label=f'Median {t}', shrink=0.7)
        _add_basemap(ax)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Hex-Density - {t}', fontsize=13)
        fig.tight_layout()
        images.append(fig_to_base64(fig))

    # Location frequency
    fig2, ax2 = plt.subplots(figsize=(12, 9))
    loc_counts = df.groupby(['Latitude', 'Longitude']).size().reset_index(name='count')
    sc2 = ax2.scatter(loc_counts['Longitude'], loc_counts['Latitude'],
                      c=loc_counts['count'], cmap='YlOrRd', s=25, alpha=0.85,
                      edgecolors='white', linewidths=0.3, zorder=5)
    plt.colorbar(sc2, ax=ax2, label='# Observations', shrink=0.7)
    _add_basemap(ax2)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title(f'Station Frequency ({len(loc_counts)} locations)', fontsize=13)
    fig2.tight_layout()
    images.append(fig_to_base64(fig2))

    # Within-location CV violin
    cv_rows = []
    for t in targets:
        grp = df.groupby(
            [df['Latitude'].round(3), df['Longitude'].round(3)]
        )[t]
        for _, g in grp:
            if len(g.dropna()) >= 3:
                cv_val = g.std() / g.mean() if g.mean() != 0 else np.nan
                if np.isfinite(cv_val):
                    cv_rows.append({'target': t, 'cv': cv_val})
    if cv_rows:
        cv_df = pd.DataFrame(cv_rows)
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        for idx_t, t in enumerate(targets):
            sub = cv_df[cv_df['target'] == t]['cv']
            parts = ax3.violinplot([sub.values], positions=[idx_t],
                                   showmedians=True, widths=0.7)
            col = TARGET_COLORS.get(t, PALETTE[idx_t])
            for pc in parts['bodies']:
                pc.set_facecolor(col)
                pc.set_alpha(0.4)
        ax3.set_xticks(range(len(targets)))
        ax3.set_xticklabels(targets, fontsize=10)
        ax3.set_ylabel('Coefficient of Variation')
        ax3.set_title('Within-Location CV per Target')
        fig3.tight_layout()
        images.append(fig_to_base64(fig3))

    basemap_note = (' Esri World Imagery tiles.' if HAS_CONTEXTILY
                    else ' <em>(Install contextily for satellite basemaps.)</em>')
    img_html = ''.join(
        f'<img src="data:image/png;base64,{i}" style="max-width:100%;margin-bottom:15px;">'
        for i in images)
    return f"""
    <h2>6 · Spatial Analysis</h2>
    <p>Hex-density maps show geographic patterns; the within-location CV violin
    reveals measurement variability.{basemap_note}</p>
    {img_html}
    """


def section_temporal(df, targets):
    if 'Sample Date' not in df.columns:
        return "<h2>7 · Temporal Analysis</h2><p>No 'Sample Date' column.</p>"

    dt = pd.to_datetime(df['Sample Date'], dayfirst=True, errors='coerce')
    df_t = df.copy()
    df_t['_dt'] = dt
    df_t['_ym'] = dt.dt.to_period('M')

    # Area chart
    fig, axes = plt.subplots(len(targets), 1,
                              figsize=(14, 4 * len(targets)), sharex=True)
    if len(targets) == 1:
        axes = [axes]
    for ax, t in zip(axes, targets):
        monthly = df_t.groupby('_ym')[t].agg(['mean', 'std']).dropna()
        x = np.arange(len(monthly))
        col = TARGET_COLORS.get(t, PALETTE[0])
        ax.fill_between(x, monthly['mean'] - monthly['std'],
                        monthly['mean'] + monthly['std'],
                        alpha=0.18, color=col)
        ax.plot(x, monthly['mean'], color=col, lw=1.8, label='Monthly mean')
        ax.set_ylabel(t)
        ax.legend(loc='upper right', fontsize=9)
        step = max(1, len(monthly) // 12)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([str(p) for p in monthly.index[::step]],
                           rotation=45, fontsize=8)
    axes[-1].set_xlabel('Month')
    fig.suptitle('Target Trends Over Time (Monthly Mean ± 1 SD)',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    img1 = fig_to_base64(fig)

    # Seasonal violin
    df_t['_month'] = dt.dt.month
    fig2, axes2 = plt.subplots(1, len(targets), figsize=(6.5 * len(targets), 5))
    if len(targets) == 1:
        axes2 = [axes2]
    for ax, t in zip(axes2, targets):
        sns.violinplot(x='_month', y=t, data=df_t, ax=ax,
                       color=TARGET_COLORS.get(t, PALETTE[0]),
                       inner='quartile', cut=0, scale='width')
        ax.set_title(t)
        ax.set_xlabel('Month')
    fig2.suptitle('Seasonal Patterns (Southern Hemisphere)', fontsize=13, y=1.02)
    fig2.tight_layout()
    img2 = fig_to_base64(fig2)

    # Year x Month heatmap
    heatmap_imgs = []
    df_t['_year'] = dt.dt.year
    for t in targets:
        pivot = df_t.pivot_table(index='_year', columns='_month',
                                  values=t, aggfunc='median')
        if pivot.shape[0] < 2:
            continue
        fig3, ax3 = plt.subplots(figsize=(10, max(4, pivot.shape[0] * 0.35)))
        sns.heatmap(pivot, cmap=TARGET_CMAPS.get(t, 'viridis'), ax=ax3,
                    linewidths=0.3, cbar_kws={'label': f'Median {t}'})
        ax3.set_title(f'{t} - Year × Month')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Year')
        fig3.tight_layout()
        heatmap_imgs.append(fig_to_base64(fig3))

    heatmap_html = ''.join(
        f'<img src="data:image/png;base64,{i}" style="max-width:100%;margin-bottom:15px;">'
        for i in heatmap_imgs)
    return f"""
    <h2>7 · Temporal Analysis</h2>
    <img src="data:image/png;base64,{img1}" style="max-width:100%">
    <h3>Seasonal Violins</h3>
    <img src="data:image/png;base64,{img2}" style="max-width:100%">
    <h3>Year × Month Heatmaps</h3>
    {heatmap_html}
    """

def section_collinearity(df, features, threshold=0.85):
    use_feats = features[:80]
    corr = df[use_feats].corr(method='spearman')

    pairs = []
    for i in range(len(use_feats)):
        for j in range(i + 1, len(use_feats)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                pairs.append((use_feats[i], use_feats[j], r))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # Dendrogram
    dist = 1 - corr.abs().values
    np.fill_diagonal(dist, 0)
    dist = np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=0.0)
    dist = np.clip(dist, 0, None)
    condensed = squareform(dist, checks=False)
    condensed = np.nan_to_num(condensed, nan=1.0, posinf=1.0, neginf=0.0)
    Z = linkage(condensed, method='average')
    fig, ax = plt.subplots(figsize=(16, max(6, len(use_feats) * 0.18)))
    dendrogram(Z, labels=use_feats, orientation='left', ax=ax,
               leaf_font_size=7, color_threshold=1 - threshold)
    ax.set_title(f'Feature Dendrogram (cut at 1−{threshold} = {1-threshold:.2f})')
    ax.set_xlabel('1 − |Spearman ρ|')
    fig.tight_layout()
    img_dend = fig_to_base64(fig)

    # Heatmap
    if len(use_feats) <= 40:
        show = use_feats
    else:
        involved = set()
        for f1, f2, _ in pairs[:50]:
            involved.update([f1, f2])
        show = sorted(involved)[:40] if len(involved) >= 3 else use_feats[:30]
    sub_corr = df[show].corr(method='spearman')
    fig2, ax2 = plt.subplots(figsize=(14, 12))
    sns.heatmap(sub_corr, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                ax=ax2, linewidths=0.1, xticklabels=True, yticklabels=True)
    ax2.tick_params(labelsize=7)
    ax2.set_title('Correlation Matrix (Spearman)')
    fig2.tight_layout()
    img_heat = fig_to_base64(fig2)

    pair_rows = ''.join(
        f'<tr><td>{f1}</td><td>{f2}</td><td>{r:.3f}</td></tr>'
        for f1, f2, r in pairs[:30])
    pair_table = (f'<table class="styled-table" border="0">'
                  f'<thead><tr><th>Feature A</th><th>Feature B</th>'
                  f'<th>Spearman ρ</th></tr></thead>'
                  f'<tbody>{pair_rows}</tbody></table>'
                  if pairs else
                  '<p>No pairs exceed the threshold - good!</p>')

    return f"""
    <h2>8 · Collinearity Diagnosis</h2>
    <p>Pairs with |ρ| ≥ {threshold}. High collinearity wastes model capacity.</p>
    <h3>Dendrogram</h3>
    <img src="data:image/png;base64,{img_dend}" style="max-width:100%">
    <h3>Heatmap</h3>
    <img src="data:image/png;base64,{img_heat}" style="max-width:100%">
    <h3>Highly Correlated Pairs (top 30)</h3>
    {pair_table}
    <p><strong>Total pairs |ρ| ≥ {threshold}:</strong> {len(pairs)}</p>
    """


def section_modellability(df, targets, features, dws_dir=None):
    X = df[features].copy()
    X_imp = _safe_impute(X)

    # Spatial CV
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        groups = (df['Latitude'].round(2).astype(str) + '_' +
                  df['Longitude'].round(2).astype(str))
        n_groups = groups.nunique()
        cv = GroupKFold(n_splits=min(5, n_groups))
        cv_args = {'groups': groups}
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_args = {}

    results = {}
    perm_data = {}
    for t in targets:
        y = df[t].copy()
        mask = y.notna()
        rf = RandomForestRegressor(n_estimators=120, max_depth=14,
                                    n_jobs=-1, random_state=42)
        scores = cross_val_score(rf, X_imp[mask], y[mask], cv=cv,
                                  scoring='r2', **cv_args)
        results[t] = {'mean_r2': scores.mean(), 'std_r2': scores.std(),
                       'fold_scores': scores}
        # Permutation importance on full fit
        rf.fit(X_imp[mask], y[mask])
        perm = permutation_importance(rf, X_imp[mask], y[mask],
                                       n_repeats=5, random_state=42, n_jobs=-1)
        perm_data[t] = pd.Series(perm.importances_mean,
                                  index=features).nlargest(15)

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results.keys())
    means = [results[t]['mean_r2'] for t in names]
    stds = [results[t]['std_r2'] for t in names]
    colors = [TARGET_COLORS.get(t, PALETTE[0]) for t in names]
    bars = ax.bar(names, means, yerr=stds, color=colors, edgecolor='white',
                  capsize=8, alpha=0.85, width=0.55)
    ax.set_ylabel('R² (Spatial CV)')
    ax.set_title('Modellability - RandomForest (120 trees, depth=14)')
    ax.set_ylim(min(0, min(means) - 0.1), 1.0)
    ax.axhline(0, color='black', lw=0.5)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.02,
                f'{m:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    fig.tight_layout()
    img_bar = fig_to_base64(fig)

    # Fold-level dot plot
    fig_fold, ax_fold = plt.subplots(figsize=(8, 4))
    for idx_t, t in enumerate(names):
        sc = results[t]['fold_scores']
        jitter = np.random.default_rng(42).normal(0, 0.04, len(sc))
        ax_fold.scatter([idx_t + j for j in jitter], sc, s=60,
                        color=TARGET_COLORS.get(t, PALETTE[0]), alpha=0.7,
                        edgecolors='white', linewidths=0.8, zorder=5)
        ax_fold.hlines(results[t]['mean_r2'], idx_t - 0.2, idx_t + 0.2,
                       colors='black', lw=2)
    ax_fold.set_xticks(range(len(names)))
    ax_fold.set_xticklabels(names, fontsize=10)
    ax_fold.set_ylabel('R²')
    ax_fold.set_title('Fold-Level R² (dots = folds, bar = mean)')
    fig_fold.tight_layout()
    img_fold = fig_to_base64(fig_fold)

    # Permutation importance
    perm_imgs = []
    for t in targets:
        s = perm_data[t]
        fig_p, ax_p = plt.subplots(figsize=(8, max(4, len(s) * 0.3)))
        y_pos = np.arange(len(s))
        ax_p.barh(y_pos, s.values, color=TARGET_COLORS.get(t, PALETTE[0]),
                  edgecolor='white', height=0.65)
        ax_p.set_yticks(y_pos)
        ax_p.set_yticklabels(s.index, fontsize=9)
        ax_p.invert_yaxis()
        ax_p.set_xlabel('Δ R² (permutation)')
        ax_p.set_title(f'Permutation Importance → {t}')
        fig_p.tight_layout()
        perm_imgs.append(fig_to_base64(fig_p))

    perm_html = ''.join(
        f'<img src="data:image/png;base64,{i}" style="max-width:100%;margin-bottom:15px;">'
        for i in perm_imgs)

    interpretations = []
    for t in names:
        r2 = results[t]['mean_r2']
        if r2 > 0.7:
            v = '🟢 Strong signal'
        elif r2 > 0.4:
            v = '🟡 Moderate signal'
        elif r2 > 0.1:
            v = '🟠 Weak signal'
        else:
            v = '🔴 Very weak / no signal'
        interpretations.append(f'<li><strong>{t}</strong> (R²={r2:.3f}): {v}</li>')

    dws_html = _subsection_dws(dws_dir, targets)

    return f"""
    <h2>9 · Modellability &amp; DWS Assessment</h2>
    <p>RandomForest with spatial cross-validation quantifies how predictable each
    target is from the available features.  Permutation importance reveals the
    true drivers.  The DWS sub-section assesses external data enrichment.</p>
    <img src="data:image/png;base64,{img_bar}" style="max-width:100%">
    <h3>Fold-Level R²</h3>
    <img src="data:image/png;base64,{img_fold}" style="max-width:100%">
    <h3>Interpretation</h3>
    <ul>{''.join(interpretations)}</ul>
    <h3>Permutation Importance</h3>
    {perm_html}
    {dws_html}
    """


def _subsection_dws(dws_dir, targets):
    if dws_dir is None:
        return ''
    try:
        from dws_data import STATION_REGISTRY, DWS_COL_MAP, load_all_station_data
    except ImportError:
        return ('<h3>DWS External Data</h3>'
                '<p class="warn-box">⚠ dws_data module not available.</p>')

    all_dws = load_all_station_data(dws_dir)
    if not all_dws:
        return ('<h3>DWS External Data</h3>'
                '<p>No DWS station data loaded.</p>')

    n_stations = len(STATION_REGISTRY)
    loaded = len(all_dws)
    total_rows = sum(len(d) for d in all_dws.values())

    # Date span
    date_ranges = []
    for stn, sdf in all_dws.items():
        if 'date' in sdf.columns and len(sdf) > 0:
            date_ranges.append({'start': sdf['date'].min(),
                                'end': sdf['date'].max()})
    dr_df = pd.DataFrame(date_ranges) if date_ranges else pd.DataFrame()
    earliest = dr_df['start'].min().strftime('%Y') if len(dr_df) > 0 else '?'
    latest = dr_df['end'].max().strftime('%Y') if len(dr_df) > 0 else '?'

    parts = [f"""
    <h3>DWS External Data Enrichment</h3>
    <p>Publicly available water quality monitoring data from the
    <a href="https://www.dws.gov.za/iwqs/wms/data/">SA Dept of Water &amp; Sanitation</a>.</p>
    <div class="stats-grid">
      <div class="stat-card"><span class="stat-num">{n_stations}</span><br>Registered Stations</div>
      <div class="stat-card"><span class="stat-num">{loaded}</span><br>Loaded</div>
      <div class="stat-card"><span class="stat-num">{total_rows:,}</span><br>Total Obs</div>
      <div class="stat-card"><span class="stat-num">{earliest}–{latest}</span><br>Date Span</div>
    </div>
    """]

    fig, ax = plt.subplots(figsize=(12, 9))
    lats = [v[0] for v in STATION_REGISTRY.values()]
    lons = [v[1] for v in STATION_REGISTRY.values()]
    sizes, colors = [], []
    for stn in STATION_REGISTRY:
        if stn in all_dws:
            n = len(all_dws[stn])
            sizes.append(max(12, min(80, n / 20)))
            colors.append(n)
        else:
            sizes.append(8)
            colors.append(0)
    sc = ax.scatter(lons, lats, c=colors, cmap='plasma', s=sizes,
                    alpha=0.85, edgecolors='white', linewidths=0.4, zorder=5)
    plt.colorbar(sc, ax=ax, label='# Observations', shrink=0.7)
    _add_basemap(ax)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'DWS Station Network ({n_stations} stations)', fontsize=13)
    fig.tight_layout()
    parts.append(f'<img src="data:image/png;base64,{fig_to_base64(fig)}" '
                 f'style="max-width:100%;margin-bottom:15px;">')

    year_data = []
    for stn, sdf in all_dws.items():
        if 'date' in sdf.columns:
            for yr, grp in sdf.groupby(sdf['date'].dt.year):
                year_data.append({'station': stn, 'year': int(yr),
                                  'count': len(grp)})
    if year_data:
        yr_df = pd.DataFrame(year_data)
        pivot = yr_df.pivot_table(index='station', columns='year',
                                  values='count', fill_value=0)
        if len(pivot) > 40:
            top_stns = yr_df.groupby('station')['count'].sum().nlargest(40).index
            pivot = pivot.loc[pivot.index.isin(top_stns)]
        fig2, ax2 = plt.subplots(figsize=(16, max(6, len(pivot) * 0.25)))
        sns.heatmap(pivot, cmap='YlGnBu', ax=ax2, linewidths=0.2,
                    cbar_kws={'label': '# Observations'})
        ax2.set_title('DWS Temporal Coverage (Station × Year)')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('')
        ax2.tick_params(labelsize=7)
        fig2.tight_layout()
        parts.append(f'<h4>Temporal Coverage</h4>'
                     f'<img src="data:image/png;base64,{fig_to_base64(fig2)}" '
                     f'style="max-width:100%;margin-bottom:15px;">')

    dws_col_map = DWS_COL_MAP
    n_maps = len(dws_col_map)
    if n_maps > 0:
        fig3, axes3 = plt.subplots(1, max(n_maps, 1),
                                    figsize=(6.5 * max(n_maps, 1), 5))
        if n_maps == 1:
            axes3 = [axes3]
        for ax, (dws_col, comp_target) in zip(axes3, dws_col_map.items()):
            dws_vals = []
            for sdf in all_dws.values():
                if dws_col in sdf.columns:
                    dws_vals.extend(
                        pd.to_numeric(sdf[dws_col], errors='coerce')
                        .dropna().tolist())
            col = TARGET_COLORS.get(comp_target, BRAND['primary'])
            if dws_vals:
                ax.hist(dws_vals, bins=60, alpha=0.45, color=col,
                        label=f'DWS ({len(dws_vals):,})',
                        edgecolor='none', density=True)
            ax.set_title(f'{comp_target}', fontsize=11)
            ax.set_xlabel('Value (DWS units)')
            ax.legend(fontsize=9)
        fig3.suptitle('DWS vs Competition Target Distributions',
                      fontsize=13, y=1.02)
        fig3.tight_layout()
        parts.append(
            f'<h4>DWS Target Distributions</h4>'
            f'<p>Distributions from DWS station records (before unit conversion).</p>'
            f'<img src="data:image/png;base64,{fig_to_base64(fig3)}" '
            f'style="max-width:100%;margin-bottom:15px;">')

    return '\n'.join(parts)


#  HTML TEMPLATE
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EDA Report - Water Quality Prediction</title>
<style>
  :root {{
    --brand:    #0B3D91;
    --accent:   #00B4D8;
    --success:  #2DC653;
    --warning:  #F77F00;
    --danger:   #D62828;
    --text:     #2B2D42;
    --bg:       #F8F9FA;
    --card-bg:  #FFFFFF;
    --shadow:   0 4px 16px rgba(0,0,0,0.07);
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 Roboto, Oxygen, sans-serif;
    max-width: 1260px; margin: 0 auto; padding: 24px 36px;
    background: var(--bg); color: var(--text);
    line-height: 1.65;
  }}
  h1 {{
    color: var(--brand); font-size: 2rem;
    border-bottom: 4px solid var(--brand); padding-bottom: 12px;
  }}
  h2 {{
    color: var(--brand); margin-top: 48px; font-size: 1.45rem;
    border-left: 5px solid var(--accent); padding-left: 14px;
  }}
  h3 {{ color: #495057; margin-top: 28px; }}
  h4 {{ color: #6C757D; }}
  img {{
    border-radius: 10px;
    box-shadow: var(--shadow);
    margin: 12px 0;
    transition: transform 0.2s;
  }}
  img:hover {{ transform: scale(1.01); }}
  .stats-grid {{
    display: flex; gap: 16px; flex-wrap: wrap; margin: 18px 0;
  }}
  .stat-card {{
    background: var(--card-bg); border-radius: 12px;
    padding: 20px 26px; box-shadow: var(--shadow);
    text-align: center; min-width: 130px; flex: 1;
    transition: box-shadow 0.2s, transform 0.2s;
  }}
  .stat-card:hover {{
    box-shadow: 0 6px 24px rgba(0,0,0,0.12);
    transform: translateY(-2px);
  }}
  .stat-num {{
    font-size: 30px; font-weight: 800; color: var(--brand);
    display: block; margin-bottom: 4px;
  }}
  .styled-table {{
    border-collapse: collapse; width: 100%; margin: 12px 0;
    background: var(--card-bg); border-radius: 10px; overflow: hidden;
    box-shadow: var(--shadow);
  }}
  .styled-table th {{
    background: var(--brand); color: white; padding: 12px 16px;
    text-align: left; font-weight: 600; font-size: 0.92rem;
  }}
  .styled-table td {{ padding: 10px 16px; border-bottom: 1px solid #EEE; }}
  .styled-table tr:hover {{ background: #F1F3F5; }}
  .badge {{
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.82rem; font-weight: 600; color: white;
  }}
  .badge-ok  {{ background: var(--success); }}
  .badge-mid {{ background: var(--warning); }}
  .badge-bad {{ background: var(--danger); }}
  .warn-box {{
    background: #FFF3CD; border-left: 4px solid var(--warning);
    padding: 12px 16px; border-radius: 6px; margin: 10px 0;
  }}
  ul {{ list-style-type: none; padding-left: 0; }}
  ul li {{ padding: 5px 0; }}
  .footer {{
    margin-top: 60px; padding: 18px; text-align: center;
    color: #ADB5BD; font-size: 0.85rem; border-top: 1px solid #DEE2E6;
  }}
  @media (max-width: 800px) {{
    body {{ padding: 12px 16px; }}
    .stats-grid {{ gap: 8px; }}
    .stat-card {{ min-width: 90px; padding: 14px 12px; }}
    .stat-num {{ font-size: 22px; }}
  }}
</style>
</head>
<body>
<h1> EDA Report - Water Quality Prediction</h1>
<p style="color:#6C757D;">Generated: {timestamp} &nbsp;|&nbsp;
Input: <code>{input_file}</code> &nbsp;|&nbsp;
Pipeline v5 - EY Open Science Data Challenge</p>
{sections}
<div class="footer">
  Auto-generated by <code>eda_report.py</code>
</div>
</body>
</html>"""


#  MAIN
def generate_report(input_path, output_path='eda_report.html', top_k=25,
                    force=False):
    if os.path.exists(output_path) and not force:
        print(f"  EDA report already exists at '{output_path}' - skipping. "
              f"Pass force=True to regenerate.")
        return

    print(f"Loading data from '{input_path}' ...")
    df = pd.read_csv(input_path)
    targets = [t for t in config.TARGETS if t in df.columns]
    if not targets:
        print(f"ERROR: No target columns found. Expected: {config.TARGETS}")
        return  
      
    features = _get_feature_cols(df, targets)
    print(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} cols  |  "
          f"{len(features)} numeric features  |  {len(targets)} targets")

    dws_dir = getattr(config, 'DWS_DIR', None)
    sections = []

    print("  [1/9] Executive summary ...")
    sections.append(_safe_section(section_overview, df, targets, features,
                                  fallback_title='1 · Executive Summary'))

    print("  [2/9] Target deep-dive ...")
    sections.append(_safe_section(section_target_distributions, df, targets,
                                  fallback_title='2 · Target Deep-Dive'))

    print("  [3/9] Feature–target correlations ...")
    sections.append(_safe_section(section_correlations, df, targets, features,
                                  top_k=top_k,
                                  fallback_title='3 · Correlations'))

    print("  [4/9] Mutual information ...")
    sections.append(_safe_section(section_mutual_info, df, targets, features,
                                  top_k=top_k,
                                  fallback_title='4 · Mutual Information'))

    print("  [5/9] Feature distributions + PCA ...")
    sections.append(_safe_section(section_feature_distributions, df, targets,
                                  features,
                                  fallback_title='5 · Feature Distributions'))

    print("  [6/9] Spatial analysis ...")
    sections.append(_safe_section(section_spatial, df, targets,
                                  fallback_title='6 · Spatial Analysis'))

    print("  [7/9] Temporal analysis ...")
    sections.append(_safe_section(section_temporal, df, targets,
                                  fallback_title='7 · Temporal Analysis'))

    print("  [8/9] Collinearity diagnosis ...")
    sections.append(_safe_section(section_collinearity, df, features,
                                  fallback_title='8 · Collinearity'))

    print("  [9/9] Modellability + DWS assessment ...")
    sections.append(_safe_section(section_modellability, df, targets, features,
                                  dws_dir=dws_dir,
                                  fallback_title='9 · Modellability & DWS'))

    html = HTML_TEMPLATE.format(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M'),
        input_file=os.path.basename(input_path),
        sections='\n'.join(sections),
    )

    with open(output_path, 'w') as f:
        f.write(html)
    size_kb = len(html) // 1024
    print(f"\n Report saved to '{output_path}' ({size_kb} KB)")
    print(f"   Open in browser: file://{os.path.abspath(output_path)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate EDA report for water quality data')
    parser.add_argument('--input', '-i',
                        default='water_quality_processed_final.csv',
                        help='Path to processed CSV')
    parser.add_argument('--output', '-o', default='eda_report.html',
                        help='Output HTML path')
    parser.add_argument('--top', '-k', type=int, default=25,
                        help='Top-K features per target')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Overwrite existing report')
    args = parser.parse_args()
    generate_report(args.input, args.output, args.top, force=args.force)
