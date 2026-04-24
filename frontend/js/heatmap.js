let heatmapData = null;

async function loadHeatmaps() {
    try {
        const response = await fetch('/api/heatmaps');
        if (!response.ok) {
            console.warn("Heatmaps data not found.");
            return;
        }
        heatmapData = await response.json();
        
        // Initial setup
        const langSelect = document.getElementById("heatmapLangSelect");
        const relSelect = document.getElementById("heatmapRelSelect");

        langSelect.addEventListener("change", () => {
            populateRelations(langSelect.value);
        });

        relSelect.addEventListener("change", () => {
            renderHeatmap(langSelect.value, relSelect.value);
        });

        // Trigger initial population
        populateRelations(langSelect.value);

    } catch (error) {
        console.error("Error loading heatmap data:", error);
    }
}

function populateRelations(lang) {
    const relSelect = document.getElementById("heatmapRelSelect");
    relSelect.innerHTML = "";

    if (!heatmapData || !heatmapData[lang]) {
        relSelect.innerHTML = '<option value="">No data</option>';
        renderHeatmap(lang, null);
        return;
    }

    const relations = Object.keys(heatmapData[lang]).sort();
    if (relations.length === 0) {
        relSelect.innerHTML = '<option value="">No data</option>';
        renderHeatmap(lang, null);
        return;
    }

    relations.forEach((rel, i) => {
        const opt = document.createElement("option");
        opt.value = rel;
        opt.textContent = rel;
        relSelect.appendChild(opt);
    });

    // Default to the first relation or 'det'/'nsubj' if available
    let defaultRel = relations[0];
    if (relations.includes("det")) defaultRel = "det";
    else if (relations.includes("nsubj")) defaultRel = "nsubj";

    relSelect.value = defaultRel;
    renderHeatmap(lang, defaultRel);
}

function renderHeatmap(lang, rel) {
    const missingEl = document.getElementById("heatmapDataMissing");
    const containerWrapper = document.getElementById("heatmapContainerWrapper");
    const chartContainer = document.getElementById("heatmapChart");

    if (!rel || !heatmapData[lang] || !heatmapData[lang][rel]) {
        missingEl.style.display = "block";
        containerWrapper.style.display = "none";
        return;
    }

    missingEl.style.display = "none";
    containerWrapper.style.display = "block";

    const data = heatmapData[lang][rel];
    const matrix = data.heatmap;
    const specialistLayer = data.specialist.layer;
    const specialistHead = data.specialist.head;

    document.getElementById("heatmapCount").textContent = data.count.toLocaleString();
    document.getElementById("heatmapSpecialist").textContent = `L${specialistLayer} H${specialistHead}`;

    chartContainer.innerHTML = "";

    const margin = { top: 30, right: 30, bottom: 40, left: 60 };
    const width = 500; 
    const height = 500;
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svg = d3.select("#heatmapChart")
        .append("svg")
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "xMidYMid meet")
        .attr("width", "100%")
        .attr("height", "auto")
        .style("max-height", "600px");

    const g = svg.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Flatten array for D3
    let flatData = [];
    let maxVal = 0;
    for (let l = 0; l < 12; l++) {
        for (let h = 0; h < 12; h++) {
            const val = matrix[l][h];
            if (val > maxVal) maxVal = val;
            flatData.push({ layer: l, head: h, value: val });
        }
    }

    const x = d3.scaleBand()
        .range([0, innerWidth])
        .domain(d3.range(12))
        .padding(0.05);

    const y = d3.scaleBand()
        .range([0, innerHeight])
        .domain(d3.range(12))
        .padding(0.05);

    // Color scale: extremely dark blue/transparent for 0, bright blue for maxVal
    const colorScale = d3.scaleLinear()
        .domain([0, maxVal])
        .range(["rgba(56, 189, 248, 0.05)", "rgba(56, 189, 248, 1)"]);

    // Axes
    g.append("g")
        .attr("transform", `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x).tickFormat(d => `H${d}`))
        .attr("color", "rgba(255,255,255,0.6)")
        .selectAll("text")
        .style("font-size", "0.75rem")
        .style("font-family", "monospace");
    
    // Label for X Axis
    svg.append("text")
        .attr("x", margin.left + innerWidth / 2)
        .attr("y", height - 5)
        .attr("text-anchor", "middle")
        .style("font-size", "0.85rem")
        .style("fill", "rgba(255,255,255,0.7)")
        .text("Attention Head");

    g.append("g")
        .call(d3.axisLeft(y).tickFormat(d => `L${d}`))
        .attr("color", "rgba(255,255,255,0.6)")
        .selectAll("text")
        .style("font-size", "0.75rem")
        .style("font-family", "monospace");

    // Label for Y Axis
    svg.append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -(margin.top + innerHeight / 2))
        .attr("y", 15)
        .attr("text-anchor", "middle")
        .style("font-size", "0.85rem")
        .style("fill", "rgba(255,255,255,0.7)")
        .text("Transformer Layer");

    // Cells
    g.selectAll(".cell")
        .data(flatData)
        .enter()
        .append("rect")
        .attr("class", "cell")
        .attr("x", d => x(d.head))
        .attr("y", d => y(d.layer))
        .attr("width", x.bandwidth())
        .attr("height", y.bandwidth())
        .style("fill", d => colorScale(d.value))
        .style("rx", 3)
        .style("ry", 3)
        .append("title")
        .text(d => `L${d.layer} H${d.head}: ${d.value.toFixed(4)}`);

    // Highlight Specialist Head
    g.selectAll(".specialist-highlight")
        .data([{ layer: specialistLayer, head: specialistHead }])
        .enter()
        .append("rect")
        .attr("x", d => x(d.head))
        .attr("y", d => y(d.layer))
        .attr("width", x.bandwidth())
        .attr("height", y.bandwidth())
        .style("fill", "none")
        .style("stroke", "#fbbf24") // Golden yellow
        .style("stroke-width", 3)
        .style("rx", 3)
        .style("ry", 3)
        .style("pointer-events", "none");

    // Max val context info
    g.append("text")
        .attr("x", innerWidth)
        .attr("y", -10)
        .attr("text-anchor", "end")
        .style("fill", "rgba(255,255,255,0.4)")
        .style("font-size", "0.75rem")
        .text(`Max Avg Attention: ${maxVal.toFixed(3)}`);
}

document.addEventListener("DOMContentLoaded", loadHeatmaps);
