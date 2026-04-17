// learn.js

let currentLang = 'asl';
// Full A-Z alphabet for the ASL dictionary (asl-labels.json only has dynamic gesture labels)
let aslVocab = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
let cslVocab = [];

async function loadVocabs() {
    try {
        const cslRes = await fetch('csl_labels.json');
        cslVocab = await cslRes.json();
        
        // Check URL for requested language
        const urlParams = new URLSearchParams(window.location.search);
        const requestedLang = urlParams.get('lang');
        if (requestedLang === 'csl') {
            switchLanguage('csl');
        } else {
            switchLanguage('asl'); // default
        }
    } catch (err) {
        console.error("Failed to load vocabularies:", err);
    }
}

window.switchLanguage = function(lang) {
    currentLang = lang;
    
    // Update tabs
    document.getElementById('tab-asl').classList.toggle('active', lang === 'asl');
    document.getElementById('tab-csl').classList.toggle('active', lang === 'csl');
    
    // Update title
    const title = document.getElementById('dictTitle');
    title.innerText = lang === 'asl' ? 'ASL Dictionary' : 'CSL Dictionary';
    
    // Update URL without reloading
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
    
    // Show CSL reference chart at the top when in CSL mode
    if (currentLang === 'csl') {
        const chartCard = document.createElement('div');
        chartCard.className = 'dict-reference-chart';
        chartCard.innerHTML = `
            <img src="images/csl/csl-reference-chart.png" alt="Khmer Sign Language Reference Chart">
        `;
        grid.parentElement.insertBefore(chartCard, grid);
    } else {
        // Remove any existing chart when switching away
        const existing = document.querySelector('.dict-reference-chart');
        if (existing) existing.remove();
    }
    
    // Also clean up chart if switching to ASL
    if (currentLang === 'asl') {
        const existing = document.querySelector('.dict-reference-chart');
        if (existing) existing.remove();
    }
    
    // Letters that use motion (GIFs work best for these)
    const dynamicSigns = ['j', 'z'];
    
    vocab.forEach(word => {
        // Skip structural words like 'idle', 'none', 'still', etc.
        const cleanWord = word.trim().toLowerCase();
        if (['idle', 'none', '', 'still'].includes(cleanWord)) return;
        
        const card = document.createElement('div');
        card.className = 'dict-card';
        
        // For dynamic signs (J, Z) try .gif first, otherwise try .jpg then .png
        const isDynamic = dynamicSigns.includes(cleanWord);
        const extensions = isDynamic ? ['gif', 'jpg', 'png'] : ['jpg', 'png', 'gif'];
        const basePath = `images/${currentLang}/${cleanWord}`;
        
        card.innerHTML = `
            <div class="dict-card-inner">
                <div class="dict-card-front">
                    <span class="flashcard-text">${word}</span>
                </div>
                <div class="dict-card-back">
                    <div class="dict-img-placeholder">
                        <img alt="${word} sign">
                        <span>Add photo for '${word}'</span>
                    </div>
                    <div class="dict-label">${word}</div>
                </div>
            </div>
        `;
        
        // Try loading each extension in order until one works
        const img = card.querySelector('img');
        let extIndex = 0;
        
        function tryNextExtension() {
            if (extIndex >= extensions.length) {
                // None of the extensions worked, show placeholder
                img.removeAttribute('src');
                return;
            }
            img.src = `${basePath}.${extensions[extIndex]}`;
            extIndex++;
        }
        
        img.onerror = tryNextExtension;
        tryNextExtension(); // Start trying
        
        grid.appendChild(card);
    });
}

// Initialize on load
document.addEventListener('DOMContentLoaded', loadVocabs);
