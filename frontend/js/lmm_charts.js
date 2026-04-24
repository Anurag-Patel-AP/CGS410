// Holds the full LMM payload once loaded
let _lmmData = null;

const LMM_LANG_META = {
    all: {
        key:      "model1",
        label:    "All Languages",
        subtitle: "Global model \u2014 all 4 languages pooled, with (1|Language) random intercept",
        color:    "#38bdf8"
    },
    en: {
        key:      "model1_en",
        label:    "English",
        subtitle: "Per-language OLS \u2014 English corpus only (y\u00a0~\u00a0C(relation)\u00a0+\u00a0sent_length)",
        color:    "#38bdf8"
    },
    hi: {
        key:      "model1_hi",
        label:    "Hindi",
        subtitle: "Per-language OLS \u2014 Hindi corpus only (y\u00a0~\u00a0C(relation)\u00a0+\u00a0sent_length)",
        color:    "#f97316"
    },
    ja: {
        key:      "model1_ja",
        label:    "Japanese",
        subtitle: "Per-language OLS \u2014 Japanese corpus only (y\u00a0~\u00a0C(relation)\u00a0+\u00a0sent_length)",
        color:    "#a78bfa"
    },
    de: {
        key:      "model1_de",
        label:    "German",
        subtitle: "Per-language OLS \u2014 German corpus only (y\u00a0~\u00a0C(relation)\u00a0+\u00a0sent_length)",
        color:    "#34d399"
    }
};

async function loadLMMResults() {
    try {
        const response = await fetch('/api/lmm_evaluation');
        if (!response.ok) {
            console.warn("LMM evaluation data not found.");
            document.getElementById("lmmContainer").innerHTML =
                "<p style='color:rgba(255,255,255,0.5); padding:1rem;'>Linear Mixed Model results not available. Please run the backend analysis script.</p>";
            return;
        }
        _lmmData = await response.json();

        const m1 = _lmmData.model1;
        const m2 = _lmmData.model2;

        document.getElementById("lmmR2_m1").textContent =
            m1.r2_conditional.toFixed(4) + " (Marginal: " + m1.r2_marginal.toFixed(4) + ")";
        document.getElementById("lmmR2_m2").textContent =
            m2.r2_conditional.toFixed(4) + " (Marginal: " + m2.r2_marginal.toFixed(4) + ")";

        // Initial render: All Languages tab
        renderLMMChart(m1.coefficients, "lmmChart", "#38bdf8");

        // Random effects (1|Language) bar chart
        if (_lmmData.random_effects) {
            renderRandomEffectsChart(_lmmData.random_effects, "lmmRandomEffectsChart");
        }

        // Wire up the tab buttons
        document.querySelectorAll(".lmm-tab").forEach(btn => {
            btn.addEventListener("click", () => {
                const lang = btn.dataset.lang;
                const meta = LMM_LANG_META[lang];
                if (!meta) return;

                const payload = _lmmData[meta.key];
                if (!payload) {
                    document.getElementById("lmmChart").innerHTML =
                        `<p style='color:rgba(255,255,255,0.4); padding:1.5rem; text-align:center;'>No per-language data for ${meta.label}.<br>Re-run <code>run_lmm_analysis.py</code> to generate per-language models.</p>`;
                    // Still update UI state
                    document.querySelectorAll(".lmm-tab").forEach(b => b.classList.remove("active"));
                    btn.classList.add("active");
                    document.getElementById("lmmChartSubtitle").textContent = meta.subtitle;
                    return;
                }

                // Switch active tab
                document.querySelectorAll(".lmm-tab").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");

                // Update subtitle
                document.getElementById("lmmChartSubtitle").textContent = meta.subtitle;

                // Re-render plot
                renderLMMChart(payload.coefficients, "lmmChart", meta.color);
            });
        });

    } catch (error) {
        console.error('Error fetching LMM results:', error);
    }
}

function renderLMMChart(coefficients, containerId, accentColor) {
    accentColor = accentColor || "#38bdf8";

    // Filter out 'Intercept', 'sent_length', 'best_head', 'best_layer'
    const relations = coefficients.filter(d =>
        d.name.startsWith("C(relation)")
    );

    // Sort by coefficient value descending
    relations.sort((a, b) => b.coef - a.coef);

    const container = document.getElementById(containerId);
    container.innerHTML = "";

    const ROW_H = 26;
    const margin = { top: 40, right: 90, bottom: 50, left: 140 };
    const width = Math.max(container.clientWidth || 700, 700);
    const height = relations.length * ROW_H + margin.top + margin.bottom;
    const innerWidth  = width  - margin.left - margin.right;
    const innerHeight = height - margin.top  - margin.bottom;

    const wrapper = d3.select("#" + containerId)
        .style("overflow-y", "auto")
        .style("overflow-x", "hidden")
        .style("-webkit-overflow-scrolling", "touch");

    const svg = wrapper
        .append("svg")
        .attr("width",   width)
        .attr("height",  height)
        .attr("display", "block");

    const g = svg.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // X scale
    const maxVal = d3.max(relations, d => Math.abs(d.coef)) || 0.1;
    const x = d3.scaleLinear()
        .domain([-maxVal - 0.05, maxVal + 0.05])
        .range([0, innerWidth]);

    // Y scale
    const y = d3.scaleBand()
        .domain(relations.map(d => d.display_name))
        .range([0, innerHeight])
        .padding(0.3);

    // X Axis
    g.append("g")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x).ticks(10))
        .attr("color", "rgba(255,255,255,0.6)");

    // Y Axis
    g.append("g")
        .call(d3.axisLeft(y))
        .attr("color", "rgba(255,255,255,0.6)")
        .selectAll("text")
        .style("font-family", "monospace")
        .style("font-size", "0.85rem");

    // Zero line
    g.append("line")
        .attr("x1", x(0)).attr("x2", x(0))
        .attr("y1", 0)   .attr("y2", innerHeight)
        .attr("stroke", "rgba(255,255,255,0.4)")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "4");

    const posColor = "#10b981";
    const negColor = "#e11d48";

    // CI lines
    g.selectAll(".ci-line")
        .data(relations).enter()
        .append("line").attr("class", "ci-line")
        .attr("x1", d => x(d.ci_low))
        .attr("x2", d => x(d.ci_high))
        .attr("y1", d => y(d.display_name) + y.bandwidth() / 2)
        .attr("y2", d => y(d.display_name) + y.bandwidth() / 2)
        .attr("stroke", d => d.coef >= 0 ? posColor : negColor)
        .attr("stroke-width", 2).attr("opacity", 0.6);

    // CI low caps
    g.selectAll(".ci-cap-low")
        .data(relations).enter()
        .append("line")
        .attr("x1", d => x(d.ci_low)).attr("x2", d => x(d.ci_low))
        .attr("y1", d => y(d.display_name) + y.bandwidth() / 2 - 4)
        .attr("y2", d => y(d.display_name) + y.bandwidth() / 2 + 4)
        .attr("stroke", d => d.coef >= 0 ? posColor : negColor)
        .attr("stroke-width", 1);

    // CI high caps
    g.selectAll(".ci-cap-high")
        .data(relations).enter()
        .append("line")
        .attr("x1", d => x(d.ci_high)).attr("x2", d => x(d.ci_high))
        .attr("y1", d => y(d.display_name) + y.bandwidth() / 2 - 4)
        .attr("y2", d => y(d.display_name) + y.bandwidth() / 2 + 4)
        .attr("stroke", d => d.coef >= 0 ? posColor : negColor)
        .attr("stroke-width", 1);

    // Dot (point estimate)
    g.selectAll(".dot")
        .data(relations).enter()
        .append("circle").attr("class", "dot")
        .attr("cx", d => x(d.coef))
        .attr("cy", d => y(d.display_name) + y.bandwidth() / 2)
        .attr("r", 5)
        .attr("fill",   d => d.coef >= 0 ? posColor : negColor)
        .attr("stroke", "#fff").attr("stroke-width", 1.5)
        .style("cursor", "pointer")
        .append("title")
        .text(d =>
            `${d.display_name}: ${d.coef.toFixed(4)}\n` +
            `95% CI: [${d.ci_low.toFixed(4)}, ${d.ci_high.toFixed(4)}]\n` +
            `P-value: ${d.pvalue.toExponential(2)}`
        );

    // Coefficient labels
    g.selectAll(".label")
        .data(relations).enter()
        .append("text").attr("class", "label")
        .attr("x", d => x(d.ci_high) + 12)
        .attr("y", d => y(d.display_name) + y.bandwidth() / 2 + 4)
        .attr("text-anchor", "start")
        .attr("fill", "rgba(255,255,255,0.7)")
        .attr("font-size", "0.75rem")
        .text(d => d.coef.toFixed(3));
}

/* ── Random Effects (1|Language) intercepts horizontal bar chart ─────────── */
function renderRandomEffectsChart(randomEffects, containerId) {
    const LANG_COLORS = {
        "English":  "#38bdf8",
        "Hindi":    "#f97316",
        "Japanese": "#a78bfa",
        "German":   "#34d399"
    };

    const container = document.getElementById(containerId);
    container.innerHTML = "";

    const sorted = [...randomEffects].sort((a, b) => b.intercept - a.intercept);

    const margin = { top: 20, right: 100, bottom: 50, left: 110 };
    const width  = Math.max(container.clientWidth || 600, 500);
    const height = sorted.length * 52 + margin.top + margin.bottom;
    const innerW = width  - margin.left - margin.right;
    const innerH = height - margin.top  - margin.bottom;

    const svg = d3.select("#" + containerId)
        .append("svg")
        .attr("width", width).attr("height", height)
        .attr("display", "block");

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const maxAbs = d3.max(sorted, d => Math.abs(d.intercept)) * 1.25 || 0.01;
    const x = d3.scaleLinear().domain([-maxAbs, maxAbs]).range([0, innerW]);
    const y = d3.scaleBand().domain(sorted.map(d => d.language)).range([0, innerH]).padding(0.35);

    // X axis
    g.append("g").attr("transform", `translate(0,${innerH})`)
        .call(d3.axisBottom(x).ticks(6).tickFormat(d => (d >= 0 ? "+" : "") + d.toFixed(4)))
        .attr("color", "rgba(255,255,255,0.5)")
        .selectAll("text").attr("fill", "rgba(255,255,255,0.6)").style("font-size", "0.75rem");

    // X label
    g.append("text").attr("x", innerW / 2).attr("y", innerH + 42)
        .attr("text-anchor", "middle").attr("fill", "rgba(255,255,255,0.5)")
        .style("font-size", "0.8rem")
        .text("Random Intercept Deviation from Global Mean");

    // Y axis
    g.append("g").call(d3.axisLeft(y))
        .attr("color", "rgba(255,255,255,0.5)")
        .selectAll("text").attr("fill", "rgba(255,255,255,0.85)").style("font-size", "0.9rem");

    // Zero line
    g.append("line")
        .attr("x1", x(0)).attr("x2", x(0))
        .attr("y1", 0)   .attr("y2", innerH)
        .attr("stroke", "rgba(255,255,255,0.35)")
        .attr("stroke-width", 1.5).attr("stroke-dasharray", "5");

    // Bars
    g.selectAll(".re-bar").data(sorted).enter()
        .append("rect").attr("class", "re-bar")
        .attr("x", d => d.intercept >= 0 ? x(0) : x(d.intercept))
        .attr("y", d => y(d.language))
        .attr("width",  d => Math.abs(x(d.intercept) - x(0)))
        .attr("height", y.bandwidth())
        .attr("fill",   d => LANG_COLORS[d.language] || "#888")
        .attr("rx", 3).attr("opacity", 0.8);

    // Value labels
    g.selectAll(".re-label").data(sorted).enter()
        .append("text").attr("class", "re-label")
        .attr("x", d => d.intercept >= 0 ? x(d.intercept) + 6 : x(d.intercept) - 6)
        .attr("y", d => y(d.language) + y.bandwidth() / 2 + 4)
        .attr("text-anchor", d => d.intercept >= 0 ? "start" : "end")
        .attr("fill", "rgba(255,255,255,0.85)")
        .style("font-size", "0.85rem").style("font-weight", "600")
        .text(d => (d.intercept >= 0 ? "+" : "") + d.intercept.toFixed(5));
}

document.addEventListener("DOMContentLoaded", loadLMMResults);
