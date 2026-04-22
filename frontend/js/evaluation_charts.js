// evaluation_charts.js
// Renders:
//  1. Bar chart — avg UAS per language (held-out test set)
//  2. Scatter/line chart — UAS vs sentence length, per language

(function () {
    const LANG_COLORS = {
        en: '#38bdf8',
        hi: '#f472b6',
        ja: '#34d399',
        de: '#fb923c'
    };
    const LANG_LABELS = {
        en: 'English', hi: 'Hindi', ja: 'Japanese', de: 'German'
    };
    const LANGS = ['en', 'hi', 'ja', 'de'];

    fetch('/api/evaluation')
        .then(r => { if (!r.ok) throw new Error(); return r.json(); })
        .then(data => {
            const valid = data.filter(d => d.uas !== null && d.uas !== undefined);
            renderAvgUAS(valid);
            renderUASvsLength(valid);
        })
        .catch(() => {
            ['avgUasChart', 'lengthUasChart'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.innerHTML = '<p style="color:#ef4444; padding:1rem;">Failed to load evaluation data.</p>';
            });
        });

    // ── Chart 1: Average UAS per Language ──────────────────────────────────
    function renderAvgUAS(data) {
        const container = document.getElementById('avgUasChart');
        if (!container) return;
        container.innerHTML = '';

        // Compute averages
        const avgData = LANGS.map(lang => {
            const vals = data.filter(d => d.language === lang).map(d => d.uas);
            const avg = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
            return { lang, name: LANG_LABELS[lang], avg: parseFloat(avg.toFixed(2)), n: vals.length };
        });

        const margin = { top: 30, right: 30, bottom: 60, left: 65 };
        const width  = Math.max(container.clientWidth || 500, 480);
        const height = 320;
        const innerW = width - margin.left - margin.right;
        const innerH = height - margin.top - margin.bottom;

        const svg = d3.select('#avgUasChart').append('svg')
            .attr('width', '100%').attr('height', height)
            .attr('viewBox', `0 0 ${width} ${height}`)
            .attr('preserveAspectRatio', 'xMinYMin meet');

        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        const xScale = d3.scaleBand().domain(avgData.map(d => d.name)).range([0, innerW]).padding(0.35);
        const yScale = d3.scaleLinear().domain([0, 70]).range([innerH, 0]);

        // Grid
        g.selectAll('.hgrid').data(yScale.ticks(5)).enter().append('line')
            .attr('x1', 0).attr('x2', innerW)
            .attr('y1', d => yScale(d)).attr('y2', d => yScale(d))
            .attr('stroke', 'rgba(255,255,255,0.07)').attr('stroke-width', 1);

        // Bars
        g.selectAll('.bar').data(avgData).enter().append('rect')
            .attr('x', d => xScale(d.name))
            .attr('y', innerH)
            .attr('width', xScale.bandwidth())
            .attr('height', 0)
            .attr('rx', 5).attr('ry', 5)
            .attr('fill', d => LANG_COLORS[d.lang])
            .attr('opacity', 0.85)
            .transition().duration(600).ease(d3.easeCubicOut).delay((_, i) => i * 80)
            .attr('y', d => yScale(d.avg))
            .attr('height', d => innerH - yScale(d.avg));

        // Value labels on top of bars
        g.selectAll('.bar-label').data(avgData).enter().append('text')
            .attr('x', d => xScale(d.name) + xScale.bandwidth() / 2)
            .attr('y', d => yScale(d.avg) - 6)
            .attr('text-anchor', 'middle')
            .attr('font-size', '12px').attr('font-weight', '700')
            .attr('font-family', 'Inter, sans-serif')
            .attr('fill', d => LANG_COLORS[d.lang])
            .attr('opacity', 0)
            .text(d => `${d.avg}%`)
            .transition().delay((_, i) => i * 80 + 400).duration(200).attr('opacity', 1);

        // n= label inside bar
        g.selectAll('.n-label').data(avgData).enter().append('text')
            .attr('x', d => xScale(d.name) + xScale.bandwidth() / 2)
            .attr('y', d => yScale(d.avg) + 18)
            .attr('text-anchor', 'middle')
            .attr('font-size', '10px').attr('fill', 'rgba(255,255,255,0.5)')
            .attr('font-family', 'Inter, sans-serif')
            .text(d => `n=${d.n}`);

        // X axis
        g.append('g').attr('transform', `translate(0,${innerH})`)
            .call(d3.axisBottom(xScale).tickSize(0))
            .call(g => g.select('.domain').attr('stroke', 'rgba(255,255,255,0.15)'))
            .selectAll('text').attr('fill', 'rgba(255,255,255,0.75)').attr('font-size', '13px').attr('dy', '1.2em');

        // Y axis
        g.append('g').call(d3.axisLeft(yScale).ticks(5).tickFormat(d => d + '%'))
            .call(g => g.select('.domain').remove())
            .selectAll('text').attr('fill', 'rgba(255,255,255,0.55)').attr('font-size', '11px');

        // Y label
        svg.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('x', -(margin.top + innerH / 2)).attr('y', 16)
            .attr('text-anchor', 'middle').attr('fill', 'rgba(255,255,255,0.4)')
            .attr('font-size', '12px').attr('font-family', 'Inter, sans-serif')
            .text('Average UAS (%)');
    }

    // ── Chart 2: UAS vs Sentence Length (scatter + smoothed line) ──────────
    function renderUASvsLength(data) {
        const container = document.getElementById('lengthUasChart');
        if (!container) return;
        container.innerHTML = '';

        const margin = { top: 30, right: 140, bottom: 55, left: 65 };
        const width  = Math.max(container.clientWidth || 700, 600);
        const height = 380;
        const innerW = width - margin.left - margin.right;
        const innerH = height - margin.top - margin.bottom;

        const maxLen = Math.max(2, Math.min(d3.max(data, d => d.length) || 30, 60));
        const minLen = Math.max(1, d3.min(data, d => d.length) || 1);

        const xScale = d3.scaleLog().domain([minLen, maxLen]).range([0, innerW]).clamp(true);
        const yScale = d3.scaleLinear().domain([0, 105]).range([innerH, 0]);

        const svg = d3.select('#lengthUasChart').append('svg')
            .attr('width', '100%').attr('height', height)
            .attr('viewBox', `0 0 ${width} ${height}`)
            .attr('preserveAspectRatio', 'xMinYMin meet');

        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        // Grid
        g.selectAll('.hgrid').data(yScale.ticks(5)).enter().append('line')
            .attr('x1', 0).attr('x2', innerW)
            .attr('y1', d => yScale(d)).attr('y2', d => yScale(d))
            .attr('stroke', 'rgba(255,255,255,0.06)').attr('stroke-width', 1);

        LANGS.forEach(lang => {
            const langData = data.filter(d => d.language === lang && d.length <= maxLen)
                                  .sort((a, b) => a.length - b.length);
            const color = LANG_COLORS[lang];

            // Scatter dots removed for cleaner graph

            // Exact lengths with a rolling window (smoothing) to start precisely at the shortest length
            const exactAvgs = {};
            langData.forEach(d => {
                const len = d.length;
                if (!exactAvgs[len]) exactAvgs[len] = { sum: 0, count: 0 };
                exactAvgs[len].sum += d.uas;
                exactAvgs[len].count++;
            });
            
            const exactData = Object.entries(exactAvgs)
                .map(([len, item]) => ({ len: +len, avg: item.sum / item.count }))
                .sort((a, b) => a.len - b.len);
                
            const lineData = exactData.filter(d => d.len <= maxLen);

            if (lineData.length > 1) {
                const lineGen = d3.line()
                    .x(d => xScale(d.len))
                    .y(d => yScale(d.avg))
                    .curve(d3.curveCatmullRom);

                g.append('path')
                    .datum(lineData)
                    .attr('fill', 'none')
                    .attr('stroke', color)
                    .attr('stroke-width', 2.5)
                    .attr('opacity', 0.9)
                    .attr('d', lineGen);
            }
        });

        // Expected value for Randomly Generated Graph
        const expectedLineData = [];
        for (let i = minLen; i <= maxLen; i++) {
            expectedLineData.push({ len: i, avg: (1 / i) * 100 });
        }
        
        const expectedLineGen = d3.line()
            .x(d => xScale(d.len))
            .y(d => yScale(d.avg));

        g.append('path')
            .datum(expectedLineData)
            .attr('fill', 'none')
            .attr('stroke', 'rgba(255, 255, 255, 0.4)')
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', '5,5')
            .attr('d', expectedLineGen);

        // X axis
        const tickVals = [1, 2, 5, 10, 20, 30, 40, 50, 60].filter(v => v >= minLen && v <= maxLen);
        g.append('g').attr('transform', `translate(0,${innerH})`)
            .call(d3.axisBottom(xScale).tickValues(tickVals).tickFormat(d3.format("d")))
            .call(g => g.select('.domain').attr('stroke', 'rgba(255,255,255,0.15)'))
            .selectAll('text').attr('fill', 'rgba(255,255,255,0.55)').attr('font-size', '11px');

        // Y axis
        g.append('g').call(d3.axisLeft(yScale).ticks(5).tickFormat(d => d + '%'))
            .call(g => g.select('.domain').remove())
            .selectAll('text').attr('fill', 'rgba(255,255,255,0.55)').attr('font-size', '11px');

        // Axis labels
        svg.append('text')
            .attr('x', margin.left + innerW / 2).attr('y', height - 8)
            .attr('text-anchor', 'middle').attr('fill', 'rgba(255,255,255,0.4)')
            .attr('font-size', '12px').attr('font-family', 'Inter, sans-serif')
            .text('Sentence Length (tokens)');

        svg.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('x', -(margin.top + innerH / 2)).attr('y', 16)
            .attr('text-anchor', 'middle').attr('fill', 'rgba(255,255,255,0.4)')
            .attr('font-size', '12px').attr('font-family', 'Inter, sans-serif')
            .text('UAS (%)');

        // Legend
        const legend = svg.append('g').attr('transform', `translate(${margin.left + innerW + 16}, ${margin.top + 10})`);
        LANGS.forEach((lang, i) => {
            const grp = legend.append('g').attr('transform', `translate(0, ${i * 24})`);
            grp.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 8).attr('y2', 8)
               .attr('stroke', LANG_COLORS[lang]).attr('stroke-width', 2.5);
            grp.append('circle').attr('cx', 9).attr('cy', 8).attr('r', 4).attr('fill', LANG_COLORS[lang]).attr('opacity', 0.5);
            grp.append('text').attr('x', 24).attr('y', 12)
               .attr('font-size', '12px').attr('font-family', 'Inter, sans-serif')
               .attr('fill', 'rgba(255,255,255,0.75)').text(LANG_LABELS[lang]);
        });

        const randomGrp = legend.append('g').attr('transform', `translate(0, ${LANGS.length * 24 + 12})`);
        randomGrp.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 8).attr('y2', 8)
            .attr('stroke', 'rgba(255,255,255,0.4)').attr('stroke-width', 2).attr('stroke-dasharray', '4,4');
        randomGrp.append('text').attr('x', 24).attr('y', 12)
            .attr('font-size', '12px').attr('font-family', 'Inter, sans-serif')
            .attr('fill', 'rgba(255,255,255,0.5)').text('E(UAS|length)');
    }
})();
