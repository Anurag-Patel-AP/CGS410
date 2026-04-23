async function loadLMMResults() {
    try {
        const response = await fetch('/api/lmm_evaluation');
        if (!response.ok) {
            console.warn("LMM evaluation data not found.");
            document.getElementById("lmmContainer").innerHTML = "<p style='color:rgba(255,255,255,0.5); padding:1rem;'>Linear Mixed Model results not available. Please run the backend analysis script.</p>";
            return;
        }
        const data = await response.json();
        
        const m1 = data.model1;
        const m2 = data.model2;
        
        document.getElementById("lmmR2_m1").textContent = m1.r2_conditional.toFixed(4) + " (Marginal: " + m1.r2_marginal.toFixed(4) + ")";
        document.getElementById("lmmR2_m2").textContent = m2.r2_conditional.toFixed(4) + " (Marginal: " + m2.r2_marginal.toFixed(4) + ")";
        
        // Fixed effects forest plot (Model 1 — proposal formula)
        renderLMMChart(m1.coefficients, "lmmChart");
        
        // Random effects (1|Language) bar chart
        if (data.random_effects) {
            renderRandomEffectsChart(data.random_effects, "lmmRandomEffectsChart");
        }
        
    } catch (error) {
        console.error('Error fetching LMM results:', error);
    }
}

function renderLMMChart(coefficients, containerId) {
    // Filter out 'Intercept', 'sent_length', 'best_head', 'best_layer'
    // Focus purely on relational coefficients
    const relations = coefficients.filter(d => 
        d.name.startsWith("C(relation)") || d.display_name !== d.name
    );
    
    // Sort by coefficient value
    relations.sort((a, b) => b.coef - a.coef);
    
    const container = document.getElementById(containerId);
    container.innerHTML = "";

    const ROW_H = 26;
    const margin = { top: 40, right: 90, bottom: 50, left: 140 };
    const width = Math.max(container.clientWidth || 700, 700);
    const height = relations.length * ROW_H + margin.top + margin.bottom;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Make wrapper scrollable vertically
    const wrapper = d3.select("#" + containerId)
        .style("overflow-y", "auto")
        .style("overflow-x", "hidden")
        .style("-webkit-overflow-scrolling", "touch");

    const svg = wrapper
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("display", "block");
        
    const g = svg.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);
        
    // X scale
    const maxVal = d3.max(relations, d => Math.abs(d.coef));
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
        
    // Center line (0)
    g.append("line")
        .attr("x1", x(0))
        .attr("x2", x(0))
        .attr("y1", 0)
        .attr("y2", innerHeight)
        .attr("stroke", "rgba(255,255,255,0.4)")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "4");

    // CI Lines (Error Bars)
    g.selectAll(".ci-line")
        .data(relations)
        .enter()
        .append("line")
        .attr("class", "ci-line")
        .attr("x1", d => x(d.ci_low))
        .attr("x2", d => x(d.ci_high))
        .attr("y1", d => y(d.display_name) + y.bandwidth() / 2)
        .attr("y2", d => y(d.display_name) + y.bandwidth() / 2)
        .attr("stroke", d => d.coef >= 0 ? "#34d399" : "#f43f5e")
        .attr("stroke-width", 2)
        .attr("opacity", 0.6);

    // End caps for CI lines
    g.selectAll(".ci-cap-low")
        .data(relations)
        .enter()
        .append("line")
        .attr("x1", d => x(d.ci_low))
        .attr("x2", d => x(d.ci_low))
        .attr("y1", d => y(d.display_name) + y.bandwidth() / 2 - 4)
        .attr("y2", d => y(d.display_name) + y.bandwidth() / 2 + 4)
        .attr("stroke", d => d.coef >= 0 ? "#34d399" : "#f43f5e")
        .attr("stroke-width", 1);

    g.selectAll(".ci-cap-high")
        .data(relations)
        .enter()
        .append("line")
        .attr("x1", d => x(d.ci_high))
        .attr("x2", d => x(d.ci_high))
        .attr("y1", d => y(d.display_name) + y.bandwidth() / 2 - 4)
        .attr("y2", d => y(d.display_name) + y.bandwidth() / 2 + 4)
        .attr("stroke", d => d.coef >= 0 ? "#34d399" : "#f43f5e")
        .attr("stroke-width", 1);

    // Dots (Point Estimates)
    g.selectAll(".dot")
        .data(relations)
        .enter()
        .append("circle")
        .attr("class", "dot")
        .attr("cx", d => x(d.coef))
        .attr("cy", d => y(d.display_name) + y.bandwidth() / 2)
        .attr("r", 5)
        .attr("fill", d => d.coef >= 0 ? "#10b981" : "#e11d48")
        .attr("stroke", "#fff")
        .attr("stroke-width", 1.5)
        .style("cursor", "pointer")
        .append("title")
        .text(d => `${d.display_name}: ${d.coef.toFixed(4)}\n95% CI: [${d.ci_low.toFixed(4)}, ${d.ci_high.toFixed(4)}]\nP-value: ${d.pvalue.toExponential(2)}`);
        
    // Labels
    g.selectAll(".label")
        .data(relations)
        .enter()
        .append("text")
        .attr("class", "label")
        .attr("x", d => x(d.ci_high) + 12)
        .attr("y", d => y(d.display_name) + y.bandwidth() / 2 + 4)
        .attr("text-anchor", "start")
        .attr("fill", "rgba(255,255,255,0.7)")
        .attr("font-size", "0.75rem")
        .text(d => d.coef.toFixed(3));
}

/* ── Random Effects (1|Language) intercepts horizontal bar chart ───────────── */
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
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const svg = d3.select("#" + containerId)
        .append("svg")
        .attr("width", width)
        .attr("height", height)
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
        .attr("y1", 0).attr("y2", innerH)
        .attr("stroke", "rgba(255,255,255,0.35)")
        .attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "5");

    // Bars
    g.selectAll(".re-bar")
        .data(sorted)
        .enter()
        .append("rect")
        .attr("class", "re-bar")
        .attr("x", d => d.intercept >= 0 ? x(0) : x(d.intercept))
        .attr("y", d => y(d.language))
        .attr("width", d => Math.abs(x(d.intercept) - x(0)))
        .attr("height", y.bandwidth())
        .attr("fill", d => LANG_COLORS[d.language] || "#888")
        .attr("rx", 3)
        .attr("opacity", 0.8);

    // Value labels at end of bars
    g.selectAll(".re-label")
        .data(sorted)
        .enter()
        .append("text")
        .attr("class", "re-label")
        .attr("x", d => d.intercept >= 0 ? x(d.intercept) + 6 : x(d.intercept) - 6)
        .attr("y", d => y(d.language) + y.bandwidth() / 2 + 4)
        .attr("text-anchor", d => d.intercept >= 0 ? "start" : "end")
        .attr("fill", "rgba(255,255,255,0.85)")
        .style("font-size", "0.85rem")
        .style("font-weight", "600")
        .text(d => (d.intercept >= 0 ? "+" : "") + d.intercept.toFixed(5));
}

document.addEventListener("DOMContentLoaded", loadLMMResults);
