// Specialist Head Bar Chart — per-language tabs
// Fetches /api/specialists/{lang} and renders a D3 horizontal bar chart

(function () {
    const LANGS = [
        { code: 'en',  label: 'English' },
        { code: 'hi',  label: 'Hindi' },
        { code: 'ja',  label: 'Japanese' },
        { code: 'de',  label: 'German' }
    ];

    let currentLang = 'en';
    const container = document.getElementById('specialistChartContainer');

    // Build tab bar
    function buildTabs() {
        const tabBar = document.createElement('div');
        tabBar.className = 'chart-tab-bar';
        tabBar.style.cssText = 'display:flex;gap:0.5rem;margin-bottom:1.5rem;flex-wrap:wrap;';

        LANGS.forEach(l => {
            const btn = document.createElement('button');
            btn.id = `tab-${l.code}`;
            btn.textContent = l.label;
            btn.style.cssText = `
                padding: 0.45rem 1.1rem;
                border-radius: 999px;
                border: 1px solid rgba(255,255,255,0.2);
                background: rgba(255,255,255,0.05);
                color: rgba(255,255,255,0.7);
                font-family: Inter, sans-serif;
                font-size: 0.85rem;
                cursor: pointer;
                transition: all 0.2s;
            `;
            btn.addEventListener('click', () => switchLang(l.code));
            tabBar.appendChild(btn);
        });
        // Clear loading placeholder
        container.innerHTML = '';
        container.appendChild(tabBar);
        // SVG will be injected here
        const chartArea = document.createElement('div');
        chartArea.id = 'specialistChartArea';
        container.appendChild(chartArea);
    }

    function setActiveTab(code) {
        LANGS.forEach(l => {
            const btn = document.getElementById(`tab-${l.code}`);
            if (!btn) return;
            if (l.code === code) {
                btn.style.background = 'var(--accent-blue, #2E86AB)';
                btn.style.color = '#fff';
                btn.style.borderColor = 'transparent';
            } else {
                btn.style.background = 'rgba(255,255,255,0.05)';
                btn.style.color = 'rgba(255,255,255,0.7)';
                btn.style.borderColor = 'rgba(255,255,255,0.2)';
            }
        });
    }

    function switchLang(code) {
        currentLang = code;
        setActiveTab(code);
        loadAndRender(code);
    }

    function loadAndRender(code) {
        const area = document.getElementById('specialistChartArea');
        area.innerHTML = '<div style="text-align:center;padding:1.5rem;color:rgba(255,255,255,0.4);">Loading...</div>';
        fetch(`/api/specialists/${code}`)
            .then(res => { if (!res.ok) throw new Error('Failed'); return res.json(); })
            .then(data => renderChart(data, area))
            .catch(() => { area.innerHTML = '<div style="text-align:center;color:#ef4444;">Failed to load data.</div>'; });
    }

    function renderChart(data, area) {
        area.innerHTML = '';

        if (!data || data.length === 0) {
            area.innerHTML = '<div style="text-align:center;color:rgba(255,255,255,0.4);">No data with sufficient support.</div>';
            return;
        }

        // Already sorted by accuracy desc from the backend JSON
        const margin = { top: 16, right: 120, bottom: 48, left: 110 };
        const barH = 30;
        const height = data.length * barH + margin.top + margin.bottom;
        const width = Math.max(area.clientWidth || 800, 600);
        const innerW = width - margin.left - margin.right;

        const colorScale = d3.scaleSequential()
            .domain([0, 100])
            .interpolator(d3.interpolateRgb('#1a3a5c', '#38bdf8'));

        const svg = d3.select(area)
            .append('svg')
            .attr('width', '100%')
            .attr('height', height)
            .attr('viewBox', `0 0 ${width} ${height}`)
            .attr('preserveAspectRatio', 'xMinYMin meet');

        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        const xScale = d3.scaleLinear().domain([0, 100]).range([0, innerW]);
        const yScale = d3.scaleBand()
            .domain(data.map(d => d.rel))
            .range([0, data.length * barH])
            .padding(0.25);

        // Grid
        g.selectAll('.grid').data(xScale.ticks(5)).enter().append('line')
            .attr('x1', d => xScale(d)).attr('x2', d => xScale(d))
            .attr('y1', 0).attr('y2', data.length * barH)
            .attr('stroke', 'rgba(255,255,255,0.07)').attr('stroke-width', 1);

        // Bars
        const bars = g.selectAll('.bar-group').data(data).enter().append('g');

        bars.append('rect')
            .attr('x', 0)
            .attr('y', d => yScale(d.rel))
            .attr('height', yScale.bandwidth())
            .attr('width', 0)
            .attr('rx', 5).attr('ry', 5)
            .attr('fill', d => colorScale(d.accuracy))
            .transition().duration(550).ease(d3.easeCubicOut)
            .delay((_, i) => i * 15)
            .attr('width', d => xScale(d.accuracy));

        // L{layer}H{head} · acc% label on bar
        bars.append('text')
            .attr('x', d => xScale(d.accuracy) + 8)
            .attr('y', d => yScale(d.rel) + yScale.bandwidth() / 2)
            .attr('dy', '0.35em')
            .attr('font-size', '11.5px')
            .attr('font-family', 'Inter, sans-serif')
            .attr('fill', 'rgba(255,255,255,0.85)')
            .attr('font-weight', '600')
            .attr('opacity', 0)
            .text(d => `L${d.layer} H${d.head}  ·  ${d.accuracy}%`)
            .transition().delay((_, i) => i * 15 + 400).duration(200)
            .attr('opacity', 1);

        // Y axis
        g.append('g').call(d3.axisLeft(yScale).tickSize(0).tickPadding(10))
            .call(g => g.select('.domain').remove())
            .selectAll('text')
            .attr('fill', 'rgba(255,255,255,0.75)')
            .attr('font-size', '12px')
            .attr('font-family', 'Inter, monospace');

        // X axis
        g.append('g')
            .attr('transform', `translate(0,${data.length * barH})`)
            .call(d3.axisBottom(xScale).ticks(5).tickFormat(d => d + '%'))
            .call(g => g.select('.domain').attr('stroke', 'rgba(255,255,255,0.15)'))
            .selectAll('text')
            .attr('fill', 'rgba(255,255,255,0.55)').attr('font-size', '11px');

        // X label
        svg.append('text')
            .attr('x', margin.left + innerW / 2).attr('y', height - 6)
            .attr('text-anchor', 'middle')
            .attr('fill', 'rgba(255,255,255,0.4)').attr('font-size', '12px')
            .attr('font-family', 'Inter, sans-serif')
            .text('Specialist Head Accuracy (%)');
    }

    // Init
    buildTabs();
    setActiveTab('en');
    loadAndRender('en');
})();
