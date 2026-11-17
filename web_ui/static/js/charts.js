/**
 * charts.js
 * =========
 * Plotly.js chart rendering logic for the Data Augmentation Web Dashboard.
 * Integrates:
 *   1. Quality vs. Diversity Trade-off Scatter Plot
 *   2. Multi-Dimensional Polar Radar Profile (Top 8 Models)
 *   3. Diversity Metrics Bar Charts
 *   4. Full Normalized Metrics Heatmap
 *   5. Exporting plots as PNG / SVG (high-res)
 *   6. Dynamic Light/Dark Theme responsiveness
 */

let currentDiversityPayload = null;

function isLightTheme() {
    return document.body.classList.contains('light-theme');
}

function getThemeLayout() {
    const isLight = isLightTheme();
    return {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {
            family: 'Inter, sans-serif',
            color: isLight ? '#334155' : '#94a3b8',
            size: 11
        },
        margin: { t: 40, r: 30, b: 50, l: 60 }
    };
}

function getPlotConfig(filename = 'benchmark_plot') {
    return {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        toImageButtonOptions: {
            format: 'png',
            filename: filename,
            height: 700,
            width: 1200,
            scale: 2.5
        }
    };
}

/**
 * Direct export trigger for saving plots as PNG or SVG
 */
function exportPlot(containerId, format = 'png', filename = 'plot') {
    const gd = document.getElementById(containerId);
    if (!gd || !gd.data) {
        alert('No chart available to export.');
        return;
    }
    Plotly.downloadImage(gd, {
        format: format,
        width: 1200,
        height: 700,
        scale: 2.5,
        filename: `${filename}_${format}`
    });
}

/**
 * 1. Trade-off Quality vs. Diversity Scatter Plot
 */
function renderTradeoffScatter(containerId, data) {
    if (!data || !data.x || data.x.length === 0) {
        document.getElementById(containerId).innerHTML = '<div class="text-center p-4">No scatter data available.</div>';
        return;
    }

    const isLight = isLightTheme();
    const hoverText = data.text.map((short, i) => {
        return `<b>${short}</b><br>` +
               `Quality Score: ${data.x[i]}<br>` +
               `Diversity Score: ${data.y[i]}<br>` +
               `Silhouette: ${data.color[i]}<br>` +
               `Generated samples: ${data.n_samples[i]}`;
    });

    const trace = {
        x: data.x,
        y: data.y,
        mode: 'markers+text',
        type: 'scatter',
        text: data.text,
        textposition: 'top right',
        textfont: { size: 9, color: isLight ? '#0f172a' : '#e2e8f0' },
        hoverinfo: 'text',
        hovertext: hoverText,
        marker: {
            size: 14,
            color: data.color,
            colorscale: 'Portland',
            showscale: true,
            colorbar: {
                title: 'Silhouette',
                thickness: 12,
                len: 0.8,
                tickfont: { color: isLight ? '#475569' : '#94a3b8', size: 10 },
                titlefont: { color: isLight ? '#0f172a' : '#cbd5e1', size: 11 }
            },
            line: { color: isLight ? '#ffffff' : '#0f172a', width: 1.5 }
        }
    };

    const layout = {
        ...getThemeLayout(),
        title: { text: 'Trade-off Quality (Fidelity) vs. Diversity', font: { color: isLight ? '#0f172a' : '#f8fafc', size: 14 } },
        xaxis: {
            title: 'Quality score (fidelity, ↑ better)',
            gridcolor: isLight ? 'rgba(0,0,0,0.07)' : 'rgba(255,255,255,0.06)',
            zerolinecolor: isLight ? 'rgba(0,0,0,0.15)' : 'rgba(255,255,255,0.1)'
        },
        yaxis: {
            title: 'Diversity score (novelty, ↑ better)',
            gridcolor: isLight ? 'rgba(0,0,0,0.07)' : 'rgba(255,255,255,0.06)',
            zerolinecolor: isLight ? 'rgba(0,0,0,0.15)' : 'rgba(255,255,255,0.1)'
        }
    };

    Plotly.newPlot(containerId, [trace], layout, getPlotConfig('tradeoff_quality_vs_diversity'));
}

/**
 * 2. Multi-Dimensional Radar Profile (Polar)
 */
function renderRadarProfile(containerId, radarData) {
    if (!radarData || !radarData.series || radarData.series.length === 0) {
        document.getElementById(containerId).innerHTML = '<div class="text-center p-4">No radar data available.</div>';
        return;
    }

    const isLight = isLightTheme();
    const categories = radarData.categories;
    const traces = radarData.series.map(item => {
        const closedCategories = [...categories, categories[0]];
        const closedValues = [...item.values, item.values[0]];

        return {
            type: 'scatterpolar',
            r: closedValues,
            theta: closedCategories,
            fill: 'toself',
            fillcolor: isLight ? 'rgba(99, 102, 241, 0.08)' : 'rgba(6, 182, 212, 0.08)',
            name: item.name,
            line: { width: 2 }
        };
    });

    const layout = {
        ...getThemeLayout(),
        polar: {
            bgcolor: isLight ? 'rgba(241, 245, 249, 0.7)' : 'rgba(15, 23, 42, 0.4)',
            radialaxis: {
                visible: true,
                range: [0, 1],
                gridcolor: isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)',
                tickfont: { color: isLight ? '#475569' : '#64748b', size: 9 }
            },
            angularaxis: {
                gridcolor: isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)',
                tickfont: { color: isLight ? '#0f172a' : '#f8fafc', size: 12, weight: 600 }
            }
        },
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.2,
            font: { color: isLight ? '#334155' : '#cbd5e1', size: 10 }
        }
    };

    Plotly.newPlot(containerId, traces, layout, getPlotConfig('radar_multicriteria_profile'));
}

/**
 * 3. Diversity Bar Chart
 */
function renderDiversityBar(containerId, payload, metricKey) {
    currentDiversityPayload = payload;
    const metricData = payload ? payload[metricKey] : null;

    if (!metricData || !metricData.algos || metricData.algos.length === 0) {
        document.getElementById(containerId).innerHTML = '<div class="text-center p-4">No diversity data available.</div>';
        return;
    }

    const isLight = isLightTheme();
    const isUp = metricData.direction === 'up';
    const trace = {
        x: metricData.algos,
        y: metricData.values,
        type: 'bar',
        marker: {
            color: metricData.values,
            colorscale: isUp ? 'Viridis' : 'Viridis_r',
            showscale: false
        }
    };

    const layout = {
        ...getThemeLayout(),
        title: { text: `${metricKey} (${isUp ? '↑ better' : '↓ better'})`, font: { color: isLight ? '#0f172a' : '#f8fafc', size: 13 } },
        xaxis: {
            tickangle: -40,
            gridcolor: isLight ? 'rgba(0,0,0,0.05)' : 'rgba(255,255,255,0.04)',
            tickfont: { size: 10, color: isLight ? '#334155' : '#94a3b8' }
        },
        yaxis: {
            title: metricKey,
            gridcolor: isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)'
        }
    };

    Plotly.newPlot(containerId, [trace], layout, getPlotConfig(`diversity_${metricKey}`));
}

/**
 * 4. Full Normalized Metrics Heatmap
 */
function renderMetricsHeatmap(containerId, heatmapData) {
    if (!heatmapData || !heatmapData.z_values || heatmapData.z_values.length === 0) {
        document.getElementById(containerId).innerHTML = '<div class="text-center p-4">No heatmap data available.</div>';
        return;
    }

    const isLight = isLightTheme();
    const trace = {
        z: heatmapData.z_values,
        x: heatmapData.x_metrics,
        y: heatmapData.y_algos,
        type: 'heatmap',
        colorscale: [
            [0.0, '#ef4444'],   // Rosso (peggiore)
            [0.4, '#f59e0b'],   // Arancione/Giallo
            [0.7, '#10b981'],   // Verde
            [1.0, '#047857']    // Verde scuro (migliore)
        ],
        hoverongaps: false,
        colorbar: {
            title: 'Normalized<br>(1=best)',
            thickness: 14,
            tickfont: { color: isLight ? '#334155' : '#94a3b8' }
        }
    };

    const layout = {
        ...getThemeLayout(),
        margin: { t: 30, r: 30, b: 120, l: 200 },
        xaxis: {
            tickangle: -45,
            gridcolor: 'transparent',
            tickfont: { size: 10, color: isLight ? '#334155' : '#cbd5e1' }
        },
        yaxis: {
            gridcolor: 'transparent',
            tickfont: { size: 10, color: isLight ? '#0f172a' : '#f1f5f9' },
            autorange: 'reversed'
        }
    };

    Plotly.newPlot(containerId, [trace], layout, getPlotConfig('heatmap_normalized_metrics'));
}
