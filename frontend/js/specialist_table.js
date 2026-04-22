// Builds the language-wise specialist head DataFrame table
// Fetches /api/specialists/{lang} for en, hi, ja, de and merges into a table

(function () {
    const LANGS = ['en', 'hi', 'ja', 'de'];
    const LANG_LABELS = { en: 'English', hi: 'Hindi', ja: 'Japanese', de: 'German' };

    Promise.all(LANGS.map(lang =>
        fetch(`/api/specialists/${lang}`)
            .then(r => r.ok ? r.json() : [])
            .then(data => ({ lang, data }))
    )).then(results => {
        // Build a map: relation → {lang: {layer, head, accuracy}}
        const relMap = {};

        results.forEach(({ lang, data }) => {
            data.forEach(d => {
                if (!relMap[d.rel]) relMap[d.rel] = {};
                relMap[d.rel][lang] = { layer: d.layer, head: d.head, accuracy: d.accuracy };
            });
        });

        // Sort relations: by average accuracy across langs that have it (desc)
        const allRels = Object.keys(relMap).sort((a, b) => {
            const avgA = Object.values(relMap[a]).reduce((s, v) => s + v.accuracy, 0) / Object.values(relMap[a]).length;
            const avgB = Object.values(relMap[b]).reduce((s, v) => s + v.accuracy, 0) / Object.values(relMap[b]).length;
            return avgB - avgA;
        });

        const tbody = document.getElementById('specialistTableBody');
        if (!tbody) return;
        tbody.innerHTML = '';

        allRels.forEach(rel => {
            const tr = document.createElement('tr');

            // Relation name
            const tdRel = document.createElement('td');
            tdRel.textContent = rel;
            tr.appendChild(tdRel);

            // Per-language cells
            LANGS.forEach(lang => {
                const td = document.createElement('td');
                const info = relMap[rel][lang];
                if (info) {
                    const span = document.createElement('span');
                    span.className = 'badge';
                    span.title = `Accuracy: ${info.accuracy}%`;
                    span.textContent = `L${info.layer} H${info.head}`;
                    td.appendChild(span);
                    td.appendChild(document.createTextNode(` ${info.accuracy}%`));
                    td.style.color = 'rgba(255,255,255,0.8)';
                    td.style.fontSize = '0.82rem';
                } else {
                    const span = document.createElement('span');
                    span.className = 'badge badge-none';
                    span.textContent = '—';
                    td.appendChild(span);
                }
                tr.appendChild(td);
            });

            tbody.appendChild(tr);
        });
    }).catch(err => {
        console.error('Failed to load specialist table:', err);
        const tbody = document.getElementById('specialistTableBody');
        if (tbody) tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:#ef4444;">Failed to load specialist data.</td></tr>';
    });
})();
