(function () {
    const LANGS = ['en', 'hi', 'ja', 'de'];
    const LANG_LABELS = { en: 'English', hi: 'Hindi', ja: 'Japanese', de: 'German' };

    Promise.all(LANGS.map(lang =>
        fetch(`/api/specialists/${lang}`)
            .then(r => r.ok ? r.json() : [])
            .then(data => ({ lang, data }))
    )).then(results => {
        const container = document.getElementById('uniqueHeadsContainer');
        if (!container) return;
        container.innerHTML = '';

        // For common heads table
        const globalHeads = {};

        results.forEach(({ lang, data }) => {
            // Find unique heads and mapping of head -> relations
            const headToRels = {};
            data.forEach(d => {
                const headKey = `L${d.layer} H${d.head}`;
                if (!headToRels[headKey]) headToRels[headKey] = [];
                headToRels[headKey].push(d.rel);

                if (!globalHeads[headKey]) globalHeads[headKey] = { langs: new Set(), rels: new Set() };
                globalHeads[headKey].langs.add(lang);
                globalHeads[headKey].rels.add(d.rel);
            });
            
            // Sort by layer then head
            const sortedHeads = Object.keys(headToRels).sort((a, b) => {
                const [lA, hA] = a.replace('L', '').split(' H').map(Number);
                const [lB, hB] = b.replace('L', '').split(' H').map(Number);
                if (lA !== lB) return lA - lB;
                return hA - hB;
            });

            // Create card
            const card = document.createElement('div');
            card.className = 'glass-panel';
            card.style.flex = '1';
            card.style.minWidth = '200px';
            card.style.padding = '1.5rem';
            card.style.background = 'rgba(255,255,255,0.02)';
            
            const title = document.createElement('h4');
            title.textContent = LANG_LABELS[lang];
            title.style.margin = '0 0 0.5rem 0';
            title.style.color = '#38bdf8';
            title.style.fontSize = '1.1rem';
            
            const count = document.createElement('p');
            count.textContent = `Unique Heads: ${sortedHeads.length} | Encodes ${data.length} Relations`;
            count.style.fontSize = '0.9rem';
            count.style.margin = '0 0 1rem 0';
            count.style.color = 'rgba(255,255,255,0.7)';

            const badgeContainer = document.createElement('div');
            badgeContainer.style.display = 'flex';
            badgeContainer.style.flexDirection = 'column'; // Stack vertically so text fits
            badgeContainer.style.gap = '0.5rem';

            sortedHeads.forEach(h => {
                const badge = document.createElement('span');
                badge.className = 'badge';
                // Inline styles based on the existing badge style
                badge.style.display = 'inline-block';
                badge.style.padding = '0.3rem 0.6rem';
                badge.style.borderRadius = '4px';
                badge.style.background = 'rgba(56,189,248,0.15)';
                badge.style.color = '#38bdf8';
                badge.style.fontSize = '0.8rem';
                badge.style.fontWeight = '600';
                
                // Add encoded relations to display
                const relsStr = headToRels[h].join(', ');
                badge.textContent = `${h} (${relsStr})`;
                badge.title = `Encodes: ${relsStr}`;
                
                badgeContainer.appendChild(badge);
            });

            card.appendChild(title);
            card.appendChild(count);
            card.appendChild(badgeContainer);
            container.appendChild(card);
        });

        // Populate common heads table
        const commonTbody = document.getElementById('commonHeadsTableBody');
        if (commonTbody) {
            commonTbody.innerHTML = '';
            // Filter heads appearing in > 1 language
            const common = Object.entries(globalHeads)
                .filter(([h, info]) => info.langs.size > 1)
                .sort((a, b) => {
                    // Sort by number of languages descending
                    if (b[1].langs.size !== a[1].langs.size) return b[1].langs.size - a[1].langs.size;
                    // Then by layer, head
                    const [lA, hA] = a[0].replace('L', '').split(' H').map(Number);
                    const [lB, hB] = b[0].replace('L', '').split(' H').map(Number);
                    if (lA !== lB) return lA - lB;
                    return hA - hB;
                });

            if (common.length === 0) {
                commonTbody.innerHTML = '<tr><td colspan="3" style="text-align:center; color:rgba(255,255,255,0.3); padding:1.5rem;">No common heads found.</td></tr>';
            } else {
                common.forEach(([h, info]) => {
                    const tr = document.createElement('tr');
                    
                    const tdHead = document.createElement('td');
                    tdHead.style.textAlign = 'left';
                    const spanHead = document.createElement('span');
                    spanHead.className = 'badge';
                    spanHead.textContent = h;
                    tdHead.appendChild(spanHead);

                    const tdLangs = document.createElement('td');
                    tdLangs.textContent = Array.from(info.langs).map(l => LANG_LABELS[l]).join(', ');
                    tdLangs.style.color = 'rgba(255,255,255,0.75)';

                    const tdRels = document.createElement('td');
                    tdRels.textContent = Array.from(info.rels).join(', ');
                    tdRels.style.color = 'rgba(255,255,255,0.75)';

                    tr.appendChild(tdHead);
                    tr.appendChild(tdLangs);
                    tr.appendChild(tdRels);
                    commonTbody.appendChild(tr);
                });
            }
        }
        
    }).catch(err => {
        console.error('Failed to load unique heads:', err);
        const container = document.getElementById('uniqueHeadsContainer');
        if (container) container.innerHTML = '<div style="color:#ef4444;">Failed to load unique heads data.</div>';
    });
})();
