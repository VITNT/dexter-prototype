let currentLang = 'asl';
const aslVocab = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('').map(word => ({ word }));
let kslVocab = ['កម្ពុជា', 'ត្រូវ', 'ឆ្ងាញ់', 'អរគុណ', 'ខុស'].map(word => ({
    word,
    basePath: `images/csl/${word}`,
    extensions: ['gif']
}));

async function loadVocabs() {
    try {
        const response = await fetch('./models/ksl/static-landmark-v1.json');
        if (response.ok) {
            const model = await response.json();
            kslVocab.push(...model.classes.map((form, index) => ({
                word: model.displayLabels[index] || form,
                form,
                basePath: `images/ksl/reference/${form}`,
                extensions: ['jpg']
            })));
        }
    } catch (error) {
        console.warn('Static KSL reference library is unavailable.', error);
    }

    const requestedLang = new URLSearchParams(window.location.search).get('lang');
    switchLanguage(requestedLang === 'csl' ? 'csl' : 'asl');
}

window.switchLanguage = function(lang) {
    currentLang = lang;
    const isAsl = lang === 'asl';

    // highlight the active language tab
    const aslTab = document.getElementById('tab-asl');
    const cslTab = document.getElementById('tab-csl');
    aslTab.classList.toggle('active', isAsl);
    cslTab.classList.toggle('active', !isAsl);
    aslTab.setAttribute('aria-pressed', String(isAsl));
    cslTab.setAttribute('aria-pressed', String(!isAsl));

    // Keep SignDex as the product name and show the selected language separately.
    const languageCode = isAsl ? 'ASL' : 'KSL';
    document.getElementById('activeLanguageCode').innerText = languageCode;
    document.title = `SignDex | ${languageCode}`;

    // secretly update the url so sharing the link works, without reloading the page
    const newUrl = new URL(window.location);
    newUrl.searchParams.set('lang', lang);
    window.history.pushState({}, '', newUrl);
    renderDictionary();
};

window.toggleFlashcardMode = function() {
    document.getElementById('dictGrid').classList.toggle(
        'flashcard-active',
        document.getElementById('flashcardToggle').checked
    );
};

function renderDictionary() {
    const grid = document.getElementById('dictGrid');
    grid.innerHTML = '';
    const vocab = currentLang === 'asl' ? aslVocab : kslVocab;
    document.querySelector('.dict-reference-chart')?.remove();

    if (currentLang === 'csl') {
        const chartCard = document.createElement('div');
        chartCard.className = 'dict-reference-chart';
        chartCard.innerHTML = '<img src="images/csl/csl-reference-chart.png" alt="KSL reference chart">';
        grid.parentElement.insertBefore(chartCard, grid);
    }

    for (const entry of vocab) {
        const word = entry.word;
        const cleanWord = word.trim().toLowerCase();
        if (['idle', 'none', '', 'still'].includes(cleanWord)) continue;

        const card = document.createElement('div');
        card.className = 'dict-card';
        card.dataset.word = cleanWord;
        if (entry.form) card.dataset.form = entry.form;
        card.addEventListener('click', (event) => {
            if (event.target.closest('.dex-nav-dropdown-item')) return;
            if (!grid.classList.contains('flashcard-active')) return;
            const wasFlipped = card.classList.contains('flipped');
            document.querySelectorAll('.dict-card.flipped').forEach(item => item.classList.remove('flipped'));
            if (!wasFlipped) card.classList.add('flipped');
        });

        card.innerHTML = `
            <div class="dict-card-inner">
                <div class="dict-card-front"><span class="flashcard-text">${word}</span></div>
                <div class="dict-card-back">
                    <div class="dict-img-placeholder loading">
                        <img alt="${word} sign">
                        <span>Reference image unavailable</span>
                    </div>
                    <div class="dict-label">${word}</div>
                </div>
            </div>`;
        grid.appendChild(card);

        const isDynamicAsl = currentLang === 'asl' && ['j', 'z'].includes(cleanWord);
        const extensions = entry.extensions || (isDynamicAsl ? ['gif', 'png', 'jpg'] : ['png', 'jpg', 'gif']);
        const basePath = entry.basePath || `images/${currentLang}/${cleanWord}`;
        loadBestImage(card, basePath, extensions);
    }
}

function loadBestImage(card, basePath, extensions) {
    const img = card.querySelector('img');
    const placeholder = card.querySelector('.dict-img-placeholder');
    let extensionIndex = 0;

    img.onload = () => {
        img.classList.add('loaded');
        placeholder.classList.remove('loading');
    };
    img.onerror = tryNextExtension;

    function tryNextExtension() {
        if (extensionIndex >= extensions.length) {
            img.removeAttribute('src');
            placeholder.classList.remove('loading');
            return;
        }
        img.src = `${basePath}.${extensions[extensionIndex++]}`;
    }
    tryNextExtension();
}

document.addEventListener('DOMContentLoaded', loadVocabs);

window.addEventListener('click', function(event) {
    const navDropdown = event.target.closest('.dex-nav-dropdown');
    const isClickInsideDropdown = event.target.closest('.dex-nav-dropdown-content');
    if (navDropdown && !isClickInsideDropdown) {
        event.preventDefault();
        navDropdown.classList.toggle('show');
    } else if (!isClickInsideDropdown) {
        document.querySelectorAll('.dex-nav-dropdown.show').forEach(dropdown => dropdown.classList.remove('show'));
    }
});
