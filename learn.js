// dictionary and flashcard page

let currentLang = 'asl';
// hardcoded A-Z for ASL since the labels file only covers dynamic gestures
let aslVocab = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
let cslVocab = [];

async function loadVocabs() {
    try {
        const cslRes = await fetch('csl_labels.json');
        cslVocab = await cslRes.json();
        
        // check if the URL says which language to show
        const urlParams = new URLSearchParams(window.location.search);
        const requestedLang = urlParams.get('lang');
        if (requestedLang === 'csl') {
            switchLanguage('csl');
        } else {
            switchLanguage('asl');
        }
    } catch (err) {
        console.error("Failed to load vocabularies:", err);
    }
}

window.switchLanguage = function(lang) {
    currentLang = lang;
    
    // highlight the right tab
    document.getElementById('tab-asl').classList.toggle('active', lang === 'asl');
    document.getElementById('tab-csl').classList.toggle('active', lang === 'csl');
    
    // update the page heading
    const title = document.getElementById('dictTitle');
    title.innerText = lang === 'asl' ? 'ASL Dictionary' : 'CSL Dictionary';
    
    // update the URL so the language persists on refresh
    const newUrl = new URL(window.location);
    newUrl.searchParams.set('lang', lang);
    window.history.pushState({}, '', newUrl);
    
    renderDictionary();
}

window.toggleFlashcardMode = function() {
    const isChecked = document.getElementById('flashcardToggle').checked;
    const grid = document.getElementById('dictGrid');
    if (isChecked) {
        grid.classList.add('flashcard-active');
    } else {
        grid.classList.remove('flashcard-active');
    }
}

function renderDictionary() {
    const grid = document.getElementById('dictGrid');
    grid.innerHTML = '';
    
    const vocab = currentLang === 'asl' ? aslVocab : cslVocab;
    
    // show the CSL reference chart above the grid when in CSL mode
    if (currentLang === 'csl') {
        const chartCard = document.createElement('div');
        chartCard.className = 'dict-reference-chart';
        chartCard.innerHTML = `
            <img src="images/csl/csl-reference-chart.png" alt="Khmer Sign Language Reference Chart">
        `;
        grid.parentElement.insertBefore(chartCard, grid);
    } else {
        // remove the chart when switching away from CSL
        const existing = document.querySelector('.dict-reference-chart');
        if (existing) existing.remove();
    }
    
    // extra cleanup just in case
    if (currentLang === 'asl') {
        const existing = document.querySelector('.dict-reference-chart');
        if (existing) existing.remove();
    }
    
    // J and Z involve motion, so GIFs show them better than static images
    const dynamicSigns = ['j', 'z'];
    
    vocab.forEach(word => {
        // skip filler labels that aren't actual signs
        const cleanWord = word.trim().toLowerCase();
        if (['idle', 'none', '', 'still'].includes(cleanWord)) return;
        
        const card = document.createElement('div');
        card.className = 'dict-card';
        
        // for J and Z try GIF first since they're animated, otherwise prefer JPG
        const isDynamic = dynamicSigns.includes(cleanWord);
        const extensions = isDynamic ? ['gif', 'jpg', 'png'] : ['jpg', 'png', 'gif'];
        const basePath = `images/${currentLang}/${cleanWord}`;
        
        card.addEventListener('click', () => {
            if (document.getElementById('dictGrid').classList.contains('flashcard-active')) {
                const wasFlipped = card.classList.contains('flipped');
                // close any other open card first
                document.querySelectorAll('.dict-card.flipped').forEach(c => c.classList.remove('flipped'));
                // then flip this one open
                if (!wasFlipped) {
                    card.classList.add('flipped');
                }
            }
        });
        
        card.innerHTML = `
            <div class="dict-card-inner">
                <div class="dict-card-front">
                    <span class="flashcard-text">${word}</span>
                </div>
                <div class="dict-card-back">
                    <div class="dict-img-placeholder">
                        <img alt="${word} sign" loading="lazy">
                        <span>Add photo for '${word}'</span>
                    </div>
                    <div class="dict-label">${word}</div>
                </div>
            </div>
        `;
        
        // try each file extension until we find one that exists
        const img = card.querySelector('img');
        let extIndex = 0;
        
        function tryNextExtension() {
            if (extIndex >= extensions.length) {
                // no image found, just show the placeholder text
                img.removeAttribute('src');
                return;
            }
            img.src = `${basePath}.${extensions[extIndex]}`;
            extIndex++;
        }
        
        img.onerror = tryNextExtension;
        // start looking for an image file
        tryNextExtension();
        
        grid.appendChild(card);
    });
}

document.addEventListener('DOMContentLoaded', loadVocabs);
