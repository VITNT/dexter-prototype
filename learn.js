// learn.js

let currentLang = 'asl';
// we use the full A-Z alphabet for the ASL dictionary since the labels file only has motion gestures
let aslVocab = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
let kslVocab = [];

async function loadVocabs() {
    try {
        const kslRes = await fetch('ksl_labels.json');
        kslVocab = await kslRes.json();
        
        // figure out which language they want based on the url
        const urlParams = new URLSearchParams(window.location.search);
        const requestedLang = urlParams.get('lang');
        if (requestedLang === 'ksl' || requestedLang === 'csl') {
            switchLanguage('ksl');
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
    document.getElementById('tab-csl').classList.toggle('active', lang === 'ksl');
    
    // swap out the dictionary title
    const title = document.getElementById('dictTitle');
    title.innerText = lang === 'asl' ? 'ASL Dictionary' : 'KSL Dictionary';
    
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
    
    const vocab = currentLang === 'asl' ? aslVocab : kslVocab;
    
    // the KSL dictionary has a helpful reference chart we show at the top
    if (currentLang === 'ksl') {
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
    
    // Create an observer to only load images when they enter the viewport
    const cardObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const card = entry.target;
                const cleanWord = card.dataset.word;
                const isDynamic = dynamicSigns.includes(cleanWord);
                const extensions = isDynamic ? ['gif', 'jpg', 'png'] : ['jpg', 'png', 'gif'];
                const basePath = `images/${currentLang === 'ksl' ? 'csl' : currentLang}/${cleanWord}`;
                
                loadBestImage(card, basePath, extensions);
                cardObserver.unobserve(card);
            }
        });
    }, { rootMargin: '200px' }); // Load images a bit before they come into view

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
        cardObserver.observe(card);
    });
}

/**
 * Tries to find the best image format in parallel and loads it
 */
async function loadBestImage(card, basePath, extensions) {
    const img = card.querySelector('img');
    const placeholder = card.querySelector('.dict-img-placeholder');
    
    // Check for existence of each extension in parallel to save time
    // We use HEAD requests to avoid downloading the full file until we pick one
    const checks = extensions.map(async (ext) => {
        try {
            const url = `${basePath}.${ext}`;
            const response = await fetch(url, { method: 'HEAD' });
            return response.ok ? url : null;
        } catch (e) {
            return null;
        }
    });

    const results = await Promise.all(checks);
    const validUrl = results.find(url => url !== null);

    if (validUrl) {
        img.src = validUrl;
        img.onload = () => {
            img.classList.add('loaded');
            placeholder.classList.remove('loading');
        };
    } else {
        // No image found, stop loading state
        placeholder.classList.remove('loading');
    }
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
