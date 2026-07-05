// learn.js

let currentLang = 'asl';
// we use the full A-Z alphabet for the ASL dictionary since the labels file only has motion gestures
let aslVocab = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
let cslVocab = ['កម្ពុជា', 'ត្រូវ', 'ឆ្ងាញ់', 'អរគុណ', 'ខុស'];

async function loadVocabs() {
    // figure out which language they want based on the url
    const urlParams = new URLSearchParams(window.location.search);
    const requestedLang = urlParams.get('lang');
    if (requestedLang === 'csl') {
        switchLanguage('csl');
    } else {
        switchLanguage('asl'); // fall back to ASL if nothing is specified
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
    const existingChart = document.querySelector('.dict-reference-chart');
    if (existingChart) existingChart.remove();
    
    // the CSL dictionary has a helpful reference chart we show at the top
    if (currentLang === 'csl') {
        const chartCard = document.createElement('div');
        chartCard.className = 'dict-reference-chart';
        chartCard.innerHTML = `
            <img src="images/csl/csl-reference-chart.png" alt="CSL Reference Chart">
        `;
        grid.parentElement.insertBefore(chartCard, grid);
    }
    
    // J and Z require motion, and every restored CSL entry is a GIF.
    const dynamicSigns = ['j', 'z'];

    vocab.forEach(word => {
        // filter out the background noise labels the model uses
        const cleanWord = word.trim().toLowerCase();
        if (['idle', 'none', '', 'still'].includes(cleanWord)) return;
        
        const card = document.createElement('div');
        card.className = 'dict-card';
        card.dataset.word = cleanWord;
        
        card.addEventListener('click', (e) => {
            // If we clicked a link inside the dropdown, don't flip the card
            if (e.target.closest('.dex-nav-dropdown-item')) return;
            
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
                    <div class="dict-img-placeholder loading">
                        <img alt="${word} sign">
                        <span>Add photo for '${word}'</span>
                    </div>
                    <div class="dict-label">${word}</div>
                </div>
            </div>
        `;
        
        grid.appendChild(card);
        const isDynamic = currentLang === 'csl' || dynamicSigns.includes(cleanWord);
        const extensions = currentLang === 'csl' ? ['gif'] : isDynamic ? ['gif', 'png', 'jpg'] : ['png', 'jpg', 'gif'];
        const basePath = `images/${currentLang}/${cleanWord}`;
        loadBestImage(card, basePath, extensions);
    });
}

/**
 * Tries image formats in order and lets the browser handle loading.
 */
function loadBestImage(card, basePath, extensions) {
    const img = card.querySelector('img');
    const placeholder = card.querySelector('.dict-img-placeholder');
    let extIndex = 0;
    
    img.onload = () => {
        img.classList.add('loaded');
        placeholder.classList.remove('loading');
    };

    img.onerror = () => {
        tryNextExtension();
    };

    function tryNextExtension() {
        if (extIndex >= extensions.length) {
            img.removeAttribute('src');
            placeholder.classList.remove('loading');
            return;
        }

        img.src = `${basePath}.${extensions[extIndex]}`;
        extIndex++;
    }

    tryNextExtension();
}

// start everything up once the page is ready
document.addEventListener('DOMContentLoaded', loadVocabs);

// Header dropdown toggle for mobile
window.addEventListener('click', function (event) {
    const navDropdown = event.target.closest('.dex-nav-dropdown');
    const isClickInsideDropdown = event.target.closest('.dex-nav-dropdown-content');

    if (navDropdown && !isClickInsideDropdown) {
        // Toggle the dropdown if clicking the 'Learn' item itself
        event.preventDefault();
        navDropdown.classList.toggle('show');
    } else if (!isClickInsideDropdown) {
        // Close all dropdowns if clicking outside
        document.querySelectorAll('.dex-nav-dropdown.show').forEach(d => d.classList.remove('show'));
    }
    // If clicking a link inside the dropdown, we let the browser handle navigation normally
});
