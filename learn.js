// learn.js

let currentLang = 'asl';
// we use the full A-Z alphabet for the ASL dictionary since the labels file only has motion gestures
let aslVocab = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
let cslVocab = [];

async function loadVocabs() {
    try {
        const cslRes = await fetch('csl_labels.json');
        cslVocab = await cslRes.json();
        
        // figure out which language they want based on the url
        const urlParams = new URLSearchParams(window.location.search);
        const requestedLang = urlParams.get('lang');
        if (requestedLang === 'csl') {
            switchLanguage('csl');
        } else {
            switchLanguage('asl'); // fall back to ASL if nothing is specified
        }
    } catch (err) {
        console.error("Failed to load vocabularies:", err);
    }
}

window.switchLanguage = function(lang) {
    currentLang = lang;
    
    // highlight the active language tab
    document.getElementById('tab-asl').classList.toggle('active', lang === 'asl');
    document.getElementById('tab-csl').classList.toggle('active', lang === 'csl');
    
    // swap out the dictionary title
    const title = document.getElementById('dictTitle');
    title.innerText = lang === 'asl' ? 'ASL Dictionary' : 'CSL Dictionary';
    
    // secretly update the url so sharing the link works, without reloading the page
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
    
    // the CSL dictionary has a helpful reference chart we show at the top
    if (currentLang === 'csl') {
        const chartCard = document.createElement('div');
        chartCard.className = 'dict-reference-chart';
        chartCard.innerHTML = `
            <img src="images/csl/csl-reference-chart.png" alt="Khmer Sign Language Reference Chart">
        `;
        grid.parentElement.insertBefore(chartCard, grid);
    } else {
        // get rid of the chart if they switch back to ASL
        const existing = document.querySelector('.dict-reference-chart');
        if (existing) existing.remove();
    }
    
    // clean up the chart just to be safe
    if (currentLang === 'asl') {
        const existing = document.querySelector('.dict-reference-chart');
        if (existing) existing.remove();
    }
    
    // J and Z require motion, so we try to load GIFs for them
    const dynamicSigns = ['j', 'z'];
    
    vocab.forEach(word => {
        // filter out the background noise labels the model uses
        const cleanWord = word.trim().toLowerCase();
        if (['idle', 'none', '', 'still'].includes(cleanWord)) return;
        
        const card = document.createElement('div');
        card.className = 'dict-card';
        
        // motion signs get gifs first, static signs get jpgs first
        const isDynamic = dynamicSigns.includes(cleanWord);
        const extensions = isDynamic ? ['gif', 'jpg', 'png'] : ['jpg', 'png', 'gif'];
        const basePath = `images/${currentLang}/${cleanWord}`;
        
        card.addEventListener('click', () => {
            if (document.getElementById('dictGrid').classList.contains('flashcard-active')) {
                const wasFlipped = card.classList.contains('flipped');
                // make sure only one card is flipped at a time
                document.querySelectorAll('.dict-card.flipped').forEach(c => c.classList.remove('flipped'));
                // flip the one they clicked
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
                        <img alt="${word} sign">
                        <span>Add photo for '${word}'</span>
                    </div>
                    <div class="dict-label">${word}</div>
                </div>
            </div>
        `;
        
        // we don't know the exact file type, so we just guess until one loads
        const img = card.querySelector('img');
        let extIndex = 0;
        
        function tryNextExtension() {
            if (extIndex >= extensions.length) {
                // we've tried everything, just show the placeholder
                img.removeAttribute('src');
                return;
            }
            img.src = `${basePath}.${extensions[extIndex]}`;
            extIndex++;
        }
        
        img.onerror = tryNextExtension;
        tryNextExtension(); // kick off the loading process
        
        grid.appendChild(card);
    });
}

// start everything up once the page is ready
document.addEventListener('DOMContentLoaded', loadVocabs);

// Header dropdown toggle for mobile
window.addEventListener('click', function (event) {
    const navDropdown = event.target.closest('.dex-nav-dropdown');
    if (navDropdown) {
        navDropdown.classList.toggle('show');
    } else {
        document.querySelectorAll('.dex-nav-dropdown.show').forEach(d => d.classList.remove('show'));
    }
});
