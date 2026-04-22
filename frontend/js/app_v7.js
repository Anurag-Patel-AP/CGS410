const API_BASE = '/api';

document.addEventListener('DOMContentLoaded', () => {

    const languageSelect = document.getElementById('languageSelect');
    const sentenceSelect = document.getElementById('sentenceSelect');
    const analyzeBtn     = document.getElementById('analyzeBtn');

    // Load Languages
    fetch(`${API_BASE}/languages`)
        .then(res => res.json())
        .then(langs => {
            languageSelect.innerHTML = '<option value="" disabled selected>Select a language</option>';
            langs.forEach(l => {
                const opt = document.createElement('option');
                opt.value = l.code;
                opt.textContent = l.name;
                languageSelect.appendChild(opt);
            });
        })
        .catch(() => {
            languageSelect.innerHTML = '<option value="" disabled selected>Failed to load API</option>';
        });

    // Language change → load sentences
    languageSelect.addEventListener('change', async (e) => {
        const lang = e.target.value;
        sentenceSelect.innerHTML = '<option value="" disabled selected>Loading sentences...</option>';
        sentenceSelect.disabled = true;
        analyzeBtn.disabled = true;

        try {
            const res = await fetch(`${API_BASE}/sentences/${lang}`);
            if (!res.ok) throw new Error();
            const sentences = await res.json();

            sentenceSelect.innerHTML = '<option value="" disabled selected>Select a sentence</option>';
            sentences.forEach((s, idx) => {
                const opt = document.createElement('option');
                opt.value = idx;
                const text = s.text.length > 80 ? s.text.substring(0, 80) + '...' : s.text;
                opt.textContent = text;
                sentenceSelect.appendChild(opt);
            });
            sentenceSelect.disabled = false;
        } catch {
            sentenceSelect.innerHTML = '<option value="" disabled selected>Error loading</option>';
        }
    });

    sentenceSelect.addEventListener('change', () => {
        if (sentenceSelect.value !== '') analyzeBtn.disabled = false;
    });

    // Analyze
    analyzeBtn.addEventListener('click', async () => {
        const lang   = languageSelect.value;
        const sentId = parseInt(sentenceSelect.value);
        if (!lang || isNaN(sentId)) return;

        analyzeBtn.disabled = true;
        document.querySelector('.btn-text').textContent = 'Analyzing...';
        document.querySelector('.loader-spinner').style.display = 'block';

        try {
            const res = await fetch(`${API_BASE}/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ language: lang, sentence_id: sentId })
            });
            if (!res.ok) throw new Error('Failed to analyze');
            const data = await res.json();

            // Show results
            const container = document.getElementById('resultsContainer');
            container.classList.remove('hidden');
            container.style.display = 'flex';

            // Language badge on model card
            const langNames = { en: 'English', hi: 'Hindi', ja: 'Japanese', de: 'German' };
            const badge = document.getElementById('modelLangBadge');
            if (badge) badge.textContent = `(${langNames[lang] || lang} specialists)`;

            // UAS
            document.getElementById('uasScore').textContent = `${data.uas}%`;

            // Convert JSON key strings → int maps
            const goldMap = {};
            for (let k in data.gold_tree) goldMap[parseInt(k)] = data.gold_tree[k];

            const predMap = {};
            for (let k in data.pred_tree) predMap[parseInt(k)] = data.pred_tree[k];

            // Render trees
            const goldViz = new TreeVisualizer('goldTreeSvg', '#2E86AB');
            const predViz = new TreeVisualizer('predTreeSvg', '#E84855');

            goldViz.draw(data.tokens, goldMap, data.deprels, null);
            predViz.draw(data.tokens, predMap, null, goldMap);

        } catch (err) {
            console.error(err);
            alert('An error occurred while analyzing the sentence.');
        } finally {
            analyzeBtn.disabled = false;
            document.querySelector('.btn-text').textContent = 'Analyze Sentence';
            document.querySelector('.loader-spinner').style.display = 'none';
        }
    });
});
