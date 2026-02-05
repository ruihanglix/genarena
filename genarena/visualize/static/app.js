/**
 * GenArena Arena Visualizer - Frontend Application
 */

// ========== State ==========
const state = {
    currentPage: 'overview', // 'overview' or 'gallery'
    subset: null,
    experiment: null,
    models: [],
    promptSources: [],
    page: 1,
    pageSize: 20,
    totalPages: 1,
    totalBattles: 0,
    filters: {
        models: [],
        result: null,
        consistent: null,
        minImages: null,
        maxImages: null,
        promptSource: null,
    },
    h2h: null,  // Head-to-head stats when 2 models selected
    imageRange: { min: 1, max: 1 },  // Available image count range for current subset
    favorites: [], // Array of {subset, exp_name, sample_index}
    viewMode: 'battles', // 'battles' or 'prompts'
    promptsPageSize: 10, // Prompts have more data, so use smaller page size
    promptsModelFilter: [], // Model filter for prompts view
    favoritesModelFilter: [], // Model filter for favorites modal
    favoritesStatsScope: 'filtered', // 'filtered' = only selected models, 'all' = all opponents
    searchQuery: '', // Search query for filtering by instruction text
    // Overview page state
    overviewData: null,
    overviewSortColumn: 'basic',
    overviewSortDirection: 'desc',
    // Cross-subset modal state
    crossSubsetState: {
        subsets: [],
        selectedSubsets: new Set(),
        subsetInfo: {},
    },
};

// ========== Model Aliases ==========
let modelAliases = {};

/**
 * Load model aliases from JSON file
 */
async function loadModelAliases() {
    try {
        const response = await fetch('static/model_aliases.json');
        if (response.ok) {
            modelAliases = await response.json();
        }
    } catch (error) {
        console.warn('Failed to load model aliases:', error);
        modelAliases = {};
    }
}

/**
 * Get display name for a model (alias if available, otherwise original name)
 * @param {string} modelName - Original model name
 * @returns {string} Display name (alias or original)
 */
function getModelDisplayName(modelName) {
    if (modelAliases[modelName] && modelAliases[modelName].alias) {
        return modelAliases[modelName].alias;
    }
    return modelName;
}

/**
 * Get model link if available
 * @param {string} modelName - Original model name
 * @returns {string|null} Link URL or null
 */
function getModelLink(modelName) {
    if (modelAliases[modelName] && modelAliases[modelName].link && modelAliases[modelName].link !== '#') {
        return modelAliases[modelName].link;
    }
    return null;
}

// ========== DOM Elements ==========
const elements = {
    // Navigation elements
    logoLink: document.getElementById('logo-link'),
    navOverview: document.getElementById('nav-overview'),
    navGallery: document.getElementById('nav-gallery'),
    // Page containers
    overviewPage: document.getElementById('overview-page'),
    galleryPage: document.getElementById('gallery-page'),
    overviewContent: document.getElementById('overview-content'),
    // Overview page elements
    crossSubsetBtn: document.getElementById('cross-subset-btn'),
    // Gallery controls
    galleryControls: document.querySelector('.gallery-controls'),
    subsetSelect: document.getElementById('subset-select'),
    expSelect: document.getElementById('exp-select'),
    // Sidebar filter elements
    modelCheckboxes: document.getElementById('model-checkboxes'),
    modelCount: document.getElementById('model-count'),
    selectAllModels: document.getElementById('select-all-models'),
    clearAllModels: document.getElementById('clear-all-models'),
    resultFilter: document.getElementById('result-filter'),
    resultFilterGroup: document.getElementById('result-filter-group'),
    consistencyFilter: document.getElementById('consistency-filter'),
    promptSourceFilterGroup: document.getElementById('prompt-source-filter-group'),
    promptSourceFilter: document.getElementById('prompt-source-filter'),
    imageCountFilterGroup: document.getElementById('image-count-filter-group'),
    minImagesSlider: document.getElementById('min-images-slider'),
    maxImagesSlider: document.getElementById('max-images-slider'),
    imageRangeDisplay: document.getElementById('image-range-display'),
    minImagesLabel: document.getElementById('min-images-label'),
    maxImagesLabel: document.getElementById('max-images-label'),
    applyFilters: document.getElementById('apply-filters'),
    clearFilters: document.getElementById('clear-filters'),
    battleList: document.getElementById('battle-list'),
    statsPanel: document.getElementById('stats-panel'),
    h2hSection: document.getElementById('h2h-section'),
    h2hPanel: document.getElementById('h2h-panel'),
    paginationInfo: document.getElementById('pagination-info'),
    pageNumbers: document.getElementById('page-numbers'),
    pageNumbersBottom: document.getElementById('page-numbers-bottom'),
    firstPage: document.getElementById('first-page'),
    prevPage: document.getElementById('prev-page'),
    nextPage: document.getElementById('next-page'),
    lastPage: document.getElementById('last-page'),
    firstPageBottom: document.getElementById('first-page-bottom'),
    prevPageBottom: document.getElementById('prev-page-bottom'),
    nextPageBottom: document.getElementById('next-page-bottom'),
    lastPageBottom: document.getElementById('last-page-bottom'),
    pageInput: document.getElementById('page-input'),
    pageGo: document.getElementById('page-go'),
    pageInputBottom: document.getElementById('page-input-bottom'),
    pageGoBottom: document.getElementById('page-go-bottom'),
    modal: document.getElementById('detail-modal'),
    modalContent: document.getElementById('detail-content'),
    modalClose: document.querySelector('.modal-close'),
    modalBackdrop: document.querySelector('.modal-backdrop'),
    lightbox: document.getElementById('lightbox'),
    lightboxImg: document.getElementById('lightbox-img'),
    lightboxLabel: document.getElementById('lightbox-label'),
    lightboxClose: document.querySelector('.lightbox-close'),
    // Favorites elements
    favoritesBtn: document.getElementById('favorites-btn'),
    favoritesCount: document.getElementById('favorites-count'),
    favoritesModal: document.getElementById('favorites-modal'),
    favoritesContent: document.getElementById('favorites-content'),
    favoritesModalClose: document.querySelector('#favorites-modal .modal-close'),
    favoritesModalBackdrop: document.querySelector('#favorites-modal .modal-backdrop'),
    clearAllFavorites: document.getElementById('clear-all-favorites'),
    // View toggle elements
    viewBattlesBtn: document.getElementById('view-battles'),
    viewPromptsBtn: document.getElementById('view-prompts'),
    promptsList: document.getElementById('prompts-list'),
    // Prompts model filter elements
    promptsModelFilterGroup: document.getElementById('prompts-model-filter-group'),
    promptsModelCheckboxes: document.getElementById('prompts-model-checkboxes'),
    promptsModelCount: document.getElementById('prompts-model-count'),
    promptsSelectAllModels: document.getElementById('prompts-select-all-models'),
    promptsClearAllModels: document.getElementById('prompts-clear-all-models'),
    promptsApplyModelFilter: document.getElementById('prompts-apply-model-filter'),
    // Favorites model filter elements
    favoritesModelFilterGroup: document.getElementById('favorites-model-filter-group'),
    favoritesModelCheckboxes: document.getElementById('favorites-model-checkboxes'),
    favoritesSelectAllModels: document.getElementById('favorites-select-all-models'),
    favoritesClearAllModels: document.getElementById('favorites-clear-all-models'),
    favoritesApplyModelFilter: document.getElementById('favorites-apply-model-filter'),
    favoritesStatsScopeAll: document.getElementById('favorites-stats-scope-all'),
    // ELO Leaderboard elements
    eloSection: document.getElementById('elo-section'),
    eloPanel: document.getElementById('elo-panel'),
    viewFullLeaderboard: document.getElementById('view-full-leaderboard'),
    leaderboardModal: document.getElementById('leaderboard-modal'),
    leaderboardContent: document.getElementById('leaderboard-content'),
    leaderboardModalClose: document.querySelector('#leaderboard-modal .modal-close'),
    leaderboardModalBackdrop: document.querySelector('#leaderboard-modal .modal-backdrop'),
    leaderboardSubsetName: document.getElementById('leaderboard-subset-name'),
    // Model Stats Modal elements
    modelStatsModal: document.getElementById('model-stats-modal'),
    modelStatsContent: document.getElementById('model-stats-content'),
    modelStatsModalClose: document.querySelector('#model-stats-modal .modal-close'),
    modelStatsModalBackdrop: document.querySelector('#model-stats-modal .modal-backdrop'),
    // Search elements
    searchInput: document.getElementById('search-input'),
    searchBtn: document.getElementById('search-btn'),
    clearSearchBtn: document.getElementById('clear-search-btn'),
    // Cross-subset modal elements
    crossSubsetModal: document.getElementById('cross-subset-modal'),
    crossSubsetModalClose: document.querySelector('#cross-subset-modal .modal-close'),
    crossSubsetModalBackdrop: document.querySelector('#cross-subset-modal .modal-backdrop'),
    crossSubsetCheckboxes: document.getElementById('cross-subset-checkboxes'),
    crossSubsetSelectAll: document.getElementById('cross-subset-select-all'),
    crossSubsetClearAll: document.getElementById('cross-subset-clear-all'),
    commonModelCount: document.getElementById('common-model-count'),
    unionModelCount: document.getElementById('union-model-count'),
    totalBattlesCount: document.getElementById('total-battles-count'),
    calculateMergedElo: document.getElementById('calculate-merged-elo'),
    crossSubsetResults: document.getElementById('cross-subset-results'),
    // Matrix modal elements
    viewMatrixBtn: document.getElementById('view-matrix'),
    matrixModal: document.getElementById('matrix-modal'),
    matrixContent: document.getElementById('matrix-content'),
    matrixSubsetName: document.getElementById('matrix-subset-name'),
    matrixModalClose: document.querySelector('#matrix-modal .modal-close'),
    matrixModalBackdrop: document.querySelector('#matrix-modal .modal-backdrop'),
    // ELO History modal elements
    viewEloHistoryBtn: document.getElementById('view-elo-history'),
    eloHistoryModal: document.getElementById('elo-history-modal'),
    eloHistoryContent: document.getElementById('elo-history-content'),
    eloHistoryLegend: document.getElementById('elo-history-legend'),
    eloHistoryGranularity: document.getElementById('elo-history-granularity'),
    eloHistoryModalClose: document.querySelector('#elo-history-modal .modal-close'),
    eloHistoryModalBackdrop: document.querySelector('#elo-history-modal .modal-backdrop'),
    // ELO by Source modal elements
    viewEloBySourceBtn: document.getElementById('view-elo-by-source'),
    eloBySourceModal: document.getElementById('elo-by-source-modal'),
    eloBySourceContent: document.getElementById('elo-by-source-content'),
    eloBySourceSubsetName: document.getElementById('elo-by-source-subset-name'),
    eloBySourceModalClose: document.querySelector('#elo-by-source-modal .modal-close'),
    eloBySourceModalBackdrop: document.querySelector('#elo-by-source-modal .modal-backdrop'),
};

// ========== API Functions ==========
async function fetchJSON(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
}

// ========== Page Navigation Functions ==========

/**
 * Switch to a different page (overview or gallery)
 */
function switchToPage(page) {
    state.currentPage = page;
    
    // Update navigation links
    elements.navOverview.classList.toggle('active', page === 'overview');
    elements.navGallery.classList.toggle('active', page === 'gallery');
    
    // Show/hide pages
    if (elements.overviewPage) {
        elements.overviewPage.style.display = page === 'overview' ? 'block' : 'none';
    }
    if (elements.galleryPage) {
        elements.galleryPage.style.display = page === 'gallery' ? 'flex' : 'none';
    }
    
    // Load page-specific data
    if (page === 'overview') {
        loadOverviewLeaderboards();
    }
    // Gallery page data is loaded when subset/experiment is selected
    
    // Update URL
    syncStateToURL();
}

/**
 * Navigate to a specific subset in the gallery
 */
function navigateToSubset(subset) {
    state.subset = subset;
    state.experiment = null;
    elements.subsetSelect.value = subset;
    switchToPage('gallery');
    loadSubsetInfo(subset);
    loadEloLeaderboard();
}

// ========== Overview Page Functions ==========

async function loadOverviewLeaderboards() {
    if (!elements.overviewContent) return;
    
    elements.overviewContent.innerHTML = '<div class="loading">Loading leaderboards...</div>';
    
    try {
        const data = await fetchJSON('api/overview/leaderboards');
        state.overviewData = data;
        renderOverviewTable();
    } catch (error) {
        console.error('Failed to load overview leaderboards:', error);
        elements.overviewContent.innerHTML = '<div class="empty-state"><p>Failed to load leaderboard data</p></div>';
    }
}

function renderOverviewTable() {
    const data = state.overviewData;
    if (!data || !data.subsets || data.subsets.length === 0) {
        elements.overviewContent.innerHTML = '<div class="empty-state"><p>No subset data available</p></div>';
        return;
    }
    
    const { subsets: rawSubsets, models, data: subsetData, subset_info } = data;
    
    // Sort subsets: basic, reasoning, multiref first, then others alphabetically
    const subsetOrder = ['basic', 'reasoning', 'multiref'];
    const subsets = [...rawSubsets].sort((a, b) => {
        const aIdx = subsetOrder.indexOf(a);
        const bIdx = subsetOrder.indexOf(b);
        if (aIdx !== -1 && bIdx !== -1) return aIdx - bIdx;
        if (aIdx !== -1) return -1;
        if (bIdx !== -1) return 1;
        return a.localeCompare(b);
    });
    
    // Sort models based on current sort settings
    const sortedModels = [...models].sort((a, b) => {
        let valA, valB;
        
        if (state.overviewSortColumn === 'model') {
            valA = a.toLowerCase();
            valB = b.toLowerCase();
            return state.overviewSortDirection === 'asc' 
                ? valA.localeCompare(valB) 
                : valB.localeCompare(valA);
        } else {
            // Sort by specific subset
            const subset = state.overviewSortColumn;
            valA = subsetData[subset]?.[a]?.elo ?? null;
            valB = subsetData[subset]?.[b]?.elo ?? null;
        }
        
        // Handle null values (put them at the end)
        if (valA === null && valB === null) return 0;
        if (valA === null) return 1;
        if (valB === null) return -1;
        
        return state.overviewSortDirection === 'asc' ? valA - valB : valB - valA;
    });
    
    // Build table header
    const sortIcon = (col) => {
        if (state.overviewSortColumn !== col) return '';
        return state.overviewSortDirection === 'asc' ? ' ▲' : ' ▼';
    };
    
    let headerHtml = `
        <th class="model-header sortable ${state.overviewSortColumn === 'model' ? 'sorted-' + state.overviewSortDirection : ''}" 
            data-sort="model">Model${sortIcon('model')}</th>
    `;
    
    subsets.forEach(subset => {
        const info = subset_info[subset] || {};
        headerHtml += `
            <th class="subset-header sortable ${state.overviewSortColumn === subset ? 'sorted-' + state.overviewSortDirection : ''}" 
                data-sort="${escapeHtml(subset)}" 
                data-subset="${escapeHtml(subset)}"
                title="Click to view ${subset} leaderboard">
                ${escapeHtml(subset)}
                <span class="subset-header-info">${info.model_count || 0} models</span>
            </th>
        `;
    });
    
    // Build table body
    let bodyHtml = '';
    sortedModels.forEach((model, idx) => {
        let rowHtml = `<td class="model-cell" data-model="${escapeHtml(model)}" title="View ${getModelDisplayName(model)} stats">`;
        
        // Add rank badge for top 3 when sorting by a subset column
        if (idx < 3 && state.overviewSortColumn !== 'model' && state.overviewSortDirection === 'desc') {
            rowHtml += `<span class="rank-badge rank-${idx + 1}">${idx + 1}</span>`;
        }
        rowHtml += `${escapeHtml(getModelDisplayName(model))}</td>`;
        
        // Add ELO for each subset
        subsets.forEach(subset => {
            const modelData = subsetData[subset]?.[model];
            if (modelData) {
                const elo = Math.round(modelData.elo);
                rowHtml += `<td class="elo-cell" title="Rank #${modelData.rank}">${elo}</td>`;
            } else {
                rowHtml += `<td class="elo-cell no-data">-</td>`;
            }
        });
        
        bodyHtml += `<tr>${rowHtml}</tr>`;
    });
    
    elements.overviewContent.innerHTML = `
        <div class="overview-table-container">
            <table class="overview-table">
                <thead><tr>${headerHtml}</tr></thead>
                <tbody>${bodyHtml}</tbody>
            </table>
        </div>
    `;
    
    // Add event listeners for sorting
    elements.overviewContent.querySelectorAll('th.sortable').forEach(th => {
        th.addEventListener('click', (e) => {
            const sortCol = th.dataset.sort;
            if (state.overviewSortColumn === sortCol) {
                state.overviewSortDirection = state.overviewSortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                state.overviewSortColumn = sortCol;
                state.overviewSortDirection = 'desc';
            }
            renderOverviewTable();
        });
    });
    
    // Add event listeners for subset header clicks (show leaderboard modal)
    elements.overviewContent.querySelectorAll('th.subset-header').forEach(th => {
        th.addEventListener('dblclick', (e) => {
            e.stopPropagation();
            const subset = th.dataset.subset;
            showSubsetLeaderboardModal(subset);
        });
    });
    
    // Add event listeners for model cell clicks
    elements.overviewContent.querySelectorAll('td.model-cell').forEach(td => {
        td.addEventListener('click', () => {
            const model = td.dataset.model;
            // Show model stats for the first subset that has this model
            const subsetWithModel = subsets.find(s => subsetData[s]?.[model]);
            if (subsetWithModel) {
                state.subset = subsetWithModel;
                loadModelStats(model);
            }
        });
    });
}

// ========== Cross-Subset Modal Functions ==========

function showCrossSubsetModal() {
    if (!elements.crossSubsetModal) return;
    elements.crossSubsetModal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
    loadCrossSubsetData();
}

function hideCrossSubsetModal() {
    if (!elements.crossSubsetModal) return;
    elements.crossSubsetModal.classList.add('hidden');
    document.body.style.overflow = '';
}

async function loadCrossSubsetData() {
    try {
        const data = await fetchJSON('api/subsets');
        state.crossSubsetState.subsets = data.subsets;
        renderCrossSubsetCheckboxes();
    } catch (error) {
        console.error('Failed to load subsets for cross-subset modal:', error);
    }
}

function renderCrossSubsetCheckboxes() {
    if (!elements.crossSubsetCheckboxes) return;
    
    const subsets = state.crossSubsetState.subsets;
    elements.crossSubsetCheckboxes.innerHTML = subsets.map(subset => `
        <div class="checkbox-item">
            <input type="checkbox" id="cross-subset-${escapeHtml(subset)}" value="${escapeHtml(subset)}"
                ${state.crossSubsetState.selectedSubsets.has(subset) ? 'checked' : ''}>
            <label for="cross-subset-${escapeHtml(subset)}">${escapeHtml(subset)}</label>
        </div>
    `).join('');
    
    // Add change listeners
    elements.crossSubsetCheckboxes.querySelectorAll('input[type="checkbox"]').forEach(cb => {
        cb.addEventListener('change', () => {
            if (cb.checked) {
                state.crossSubsetState.selectedSubsets.add(cb.value);
            } else {
                state.crossSubsetState.selectedSubsets.delete(cb.value);
            }
            updateCrossSubsetInfo();
        });
    });
}

async function updateCrossSubsetInfo() {
    const selected = Array.from(state.crossSubsetState.selectedSubsets);
    
    if (selected.length === 0) {
        elements.commonModelCount.textContent = '-';
        elements.unionModelCount.textContent = '-';
        elements.totalBattlesCount.textContent = '-';
        return;
    }
    
    try {
        const data = await fetchJSON(`api/cross-subset/info?subsets=${selected.join(',')}`);
        state.crossSubsetState.subsetInfo = data;
        
        elements.commonModelCount.textContent = data.common_models?.length || 0;
        elements.unionModelCount.textContent = data.all_models?.length || 0;
        elements.totalBattlesCount.textContent = data.total_battles || 0;
    } catch (error) {
        console.error('Failed to load cross-subset info:', error);
    }
}

async function calculateMergedEloForPage() {
    const selected = Array.from(state.crossSubsetState.selectedSubsets);
    
    if (selected.length === 0) {
        alert('Please select at least one subset');
        return;
    }
    
    const modelScope = document.querySelector('input[name="model-scope"]:checked')?.value || 'all';
    
    if (!elements.crossSubsetResults) return;
    elements.crossSubsetResults.innerHTML = '<div class="loading">Calculating merged ELO...</div>';
    
    try {
        const data = await fetchJSON(`api/cross-subset/elo?subsets=${selected.join(',')}&model_scope=${modelScope}`);
        renderCrossSubsetResults(data);
    } catch (error) {
        console.error('Failed to calculate merged ELO:', error);
        elements.crossSubsetResults.innerHTML = '<div class="empty-state"><p>Failed to calculate merged ELO</p></div>';
    }
}

function renderCrossSubsetResults(data) {
    if (!elements.crossSubsetResults) return;
    
    const { leaderboard, subsets, total_battles } = data;
    
    if (!leaderboard || leaderboard.length === 0) {
        elements.crossSubsetResults.innerHTML = '<div class="empty-state"><p>No results available</p></div>';
        return;
    }
    
    const tableRows = leaderboard.map((model, idx) => {
        const rank = idx + 1;
        const rankClass = rank <= 3 ? `rank-${rank}` : '';
        const winRatePercent = (model.win_rate * 100).toFixed(1);
        
        return `
            <tr>
                <td class="rank-cell ${rankClass}">#${rank}</td>
                <td class="model-cell">${escapeHtml(getModelDisplayName(model.model))}</td>
                <td class="elo-cell">${Math.round(model.elo)}</td>
                <td class="stat-cell wins">${model.wins}</td>
                <td class="stat-cell losses">${model.losses}</td>
                <td class="stat-cell ties">${model.ties}</td>
                <td class="stat-cell">${model.total || (model.wins + model.losses + model.ties)}</td>
                <td class="win-rate-cell">${winRatePercent}%</td>
            </tr>
        `;
    }).join('');
    
    elements.crossSubsetResults.innerHTML = `
        <h3>Merged ELO Results</h3>
        <div class="merged-elo-info">
            <p>Combined ELO from ${subsets.length} subset(s): ${escapeHtml(subsets.join(', '))}</p>
            <p>Total battles: ${total_battles}</p>
        </div>
        <table class="merged-leaderboard">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>ELO</th>
                    <th>Wins</th>
                    <th>Losses</th>
                    <th>Ties</th>
                    <th>Total</th>
                    <th>Win %</th>
                </tr>
            </thead>
            <tbody>${tableRows}</tbody>
        </table>
    `;
}

// ========== Subset Data Loading Functions ==========

async function loadSubsets() {
    try {
        const data = await fetchJSON('api/subsets');
        elements.subsetSelect.innerHTML = '<option value="">Select subset...</option>';
        data.subsets.forEach(subset => {
            const option = document.createElement('option');
            option.value = subset;
            option.textContent = subset;
            elements.subsetSelect.appendChild(option);
        });
        return data;
    } catch (error) {
        console.error('Failed to load subsets:', error);
        return { subsets: [] };
    }
}

async function loadSubsetInfo(subset) {
    try {
        const data = await fetchJSON(`api/subsets/${subset}/info`);

        // Sort experiments by date suffix (descending - newest first)
        // Format: xxx_yyyymmdd
        const sortedExperiments = [...data.experiments].sort((a, b) => {
            const dateA = a.match(/_(\d{8})$/)?.[1] || '00000000';
            const dateB = b.match(/_(\d{8})$/)?.[1] || '00000000';
            return dateB.localeCompare(dateA);  // Descending order
        });

        // Update experiments dropdown
        elements.expSelect.innerHTML = '<option value="">Select experiment...</option>';
        // Add "Show All" option (always show if there are experiments)
        if (sortedExperiments.length >= 1) {
            const allOption = document.createElement('option');
            allOption.value = '__all__';
            allOption.textContent = '📊 Show All';
            elements.expSelect.appendChild(allOption);
        }
        sortedExperiments.forEach(exp => {
            const option = document.createElement('option');
            option.value = exp;
            // Non-GenArena_ experiments get indented with two non-breaking spaces
            const isGenArena = exp.startsWith('GenArena_');
            option.textContent = isGenArena ? exp : '\u00A0\u00A0' + exp;
            elements.expSelect.appendChild(option);
        });
        elements.expSelect.disabled = false;

        // Update model checkboxes
        state.models = data.models;
        renderModelCheckboxes(data.models);

        // Also update prompts model filter checkboxes
        renderPromptsModelCheckboxes();

        // Update prompt source filter
        state.promptSources = data.prompt_sources || [];
        updatePromptSourceFilter(state.promptSources);

        // Update image range filter
        state.imageRange = {
            min: data.min_input_images || 1,
            max: data.max_input_images || 1
        };
        updateImageRangeSlider();

    } catch (error) {
        console.error('Failed to load subset info:', error);
    }
}

function updatePromptSourceFilter(sources) {
    elements.promptSourceFilter.innerHTML = '<option value="">All sources</option>';

    if (sources.length > 0) {
        sources.forEach(source => {
            const option = document.createElement('option');
            option.value = source;
            option.textContent = source;
            elements.promptSourceFilter.appendChild(option);
        });
        elements.promptSourceFilterGroup.style.display = 'block';
    } else {
        elements.promptSourceFilterGroup.style.display = 'none';
    }
}

function renderModelCheckboxes(models) {
    elements.modelCheckboxes.innerHTML = models.map(model => `
        <div class="checkbox-item">
            <input type="checkbox" id="model-${escapeHtml(model)}" value="${escapeHtml(model)}">
            <label for="model-${escapeHtml(model)}">${escapeHtml(getModelDisplayName(model))}</label>
        </div>
    `).join('');

    // Add change listeners
    elements.modelCheckboxes.querySelectorAll('input[type="checkbox"]').forEach(cb => {
        cb.addEventListener('change', updateModelSelection);
    });

    updateModelCount();
}

function updateModelSelection() {
    const selected = getSelectedModels();
    updateModelCount();

    // Update result filter based on selection
    if (selected.length === 1) {
        // Single model: show wins/losses/ties
        elements.resultFilterGroup.style.display = 'block';
        elements.resultFilter.innerHTML = `
            <option value="">All results</option>
            <option value="wins">Wins</option>
            <option value="losses">Losses</option>
            <option value="ties">Ties</option>
        `;
    } else if (selected.length === 2) {
        // Two models: show filter by winner
        elements.resultFilterGroup.style.display = 'block';
        elements.resultFilter.innerHTML = `
            <option value="">All results</option>
            <option value="${escapeHtml(selected[0])}">${escapeHtml(selected[0])} wins</option>
            <option value="${escapeHtml(selected[1])}">${escapeHtml(selected[1])} wins</option>
            <option value="ties">Ties</option>
        `;
    } else {
        elements.resultFilterGroup.style.display = 'none';
        elements.resultFilter.value = '';
    }
}

function getSelectedModels() {
    const checkboxes = elements.modelCheckboxes.querySelectorAll('input[type="checkbox"]:checked');
    return Array.from(checkboxes).map(cb => cb.value);
}

function updateModelCount() {
    const count = getSelectedModels().length;
    elements.modelCount.textContent = `(${count} selected)`;
}

function updateImageRangeSlider() {
    const { min, max } = state.imageRange;

    // Show/hide the filter based on whether there's a range
    if (min === max) {
        elements.imageCountFilterGroup.style.display = 'none';
        return;
    }

    elements.imageCountFilterGroup.style.display = 'block';

    // Update slider attributes
    elements.minImagesSlider.min = min;
    elements.minImagesSlider.max = max;
    elements.minImagesSlider.value = min;

    elements.maxImagesSlider.min = min;
    elements.maxImagesSlider.max = max;
    elements.maxImagesSlider.value = max;

    // Update labels
    elements.minImagesLabel.textContent = min;
    elements.maxImagesLabel.textContent = max;
    updateImageRangeDisplay();
}

function updateImageRangeDisplay() {
    const minVal = parseInt(elements.minImagesSlider.value);
    const maxVal = parseInt(elements.maxImagesSlider.value);
    elements.imageRangeDisplay.textContent = `${minVal}-${maxVal}`;
}

async function loadH2HStats() {
    const models = state.filters.models;
    if (models.length !== 2 || !state.subset || !state.experiment) {
        elements.h2hSection.style.display = 'none';
        state.h2h = null;
        return;
    }

    try {
        const url = `api/subsets/${state.subset}/experiments/${state.experiment}/h2h?model_a=${encodeURIComponent(models[0])}&model_b=${encodeURIComponent(models[1])}`;
        const data = await fetchJSON(url);
        state.h2h = data;
        renderH2HStats(data);
        elements.h2hSection.style.display = 'block';
    } catch (error) {
        console.error('Failed to load H2H stats:', error);
        elements.h2hSection.style.display = 'none';
    }
}

function renderH2HStats(h2h) {
    const winRateA = (h2h.win_rate_a * 100).toFixed(1);
    const winRateB = (h2h.win_rate_b * 100).toFixed(1);
    const tieRate = (h2h.tie_rate * 100).toFixed(1);

    // Calculate bar widths
    const total = h2h.wins_a + h2h.wins_b + h2h.ties;
    const widthA = total > 0 ? (h2h.wins_a / total * 100) : 0;
    const widthTie = total > 0 ? (h2h.ties / total * 100) : 0;
    const widthB = total > 0 ? (h2h.wins_b / total * 100) : 0;

    elements.h2hPanel.innerHTML = `
        <div class="h2h-labels">
            <span class="h2h-label" title="${escapeHtml(h2h.model_a)}">${escapeHtml(getModelDisplayName(h2h.model_a))}</span>
            <span class="h2h-label" title="${escapeHtml(h2h.model_b)}">${escapeHtml(getModelDisplayName(h2h.model_b))}</span>
        </div>
        <div class="h2h-bar">
            ${widthA > 0 ? `<div class="h2h-bar-a" style="width: ${widthA}%">${h2h.wins_a}</div>` : ''}
            ${widthTie > 0 ? `<div class="h2h-bar-tie" style="width: ${widthTie}%">${h2h.ties}</div>` : ''}
            ${widthB > 0 ? `<div class="h2h-bar-b" style="width: ${widthB}%">${h2h.wins_b}</div>` : ''}
        </div>
        <div class="h2h-stats-row">
            <span>Total battles</span>
            <span class="value">${h2h.total}</span>
        </div>
        <div class="h2h-stats-row">
            <span>Win rate</span>
            <span class="value">${winRateA}% / ${tieRate}% / ${winRateB}%</span>
        </div>
    `;
}

async function loadBattles() {
    if (!state.subset || !state.experiment) {
        return;
    }

    // Build URL with filters
    const params = new URLSearchParams({
        page: state.page,
        page_size: state.pageSize,
    });

    if (state.filters.models && state.filters.models.length > 0) {
        params.append('models', state.filters.models.join(','));
    }
    if (state.filters.result) {
        params.append('result', state.filters.result);
    }
    if (state.filters.consistent !== null) {
        params.append('consistent', state.filters.consistent);
    }
    if (state.filters.minImages !== null) {
        params.append('min_images', state.filters.minImages);
    }
    if (state.filters.maxImages !== null) {
        params.append('max_images', state.filters.maxImages);
    }
    if (state.filters.promptSource) {
        params.append('prompt_source', state.filters.promptSource);
    }

    // Use search API if there's a search query
    let url;
    if (state.searchQuery) {
        params.append('q', state.searchQuery);
        url = `api/subsets/${state.subset}/experiments/${state.experiment}/search?${params}`;
    } else {
        url = `api/subsets/${state.subset}/experiments/${state.experiment}/battles?${params}`;
    }

    // Show loading state
    elements.battleList.innerHTML = '<div class="loading">Loading battles</div>';

    try {
        const data = await fetchJSON(url);

        state.totalPages = data.total_pages;
        state.totalBattles = data.total;

        renderBattles(data.battles);
        updatePagination();
        updateStats();
        loadH2HStats();

    } catch (error) {
        console.error('Failed to load battles:', error);
        elements.battleList.innerHTML = '<div class="empty-state"><p>Failed to load battles</p></div>';
    }
}

async function loadPrompts() {
    console.log('loadPrompts called, state:', { subset: state.subset, experiment: state.experiment, viewMode: state.viewMode });
    if (!state.subset || !state.experiment) {
        console.log('loadPrompts: subset or experiment not selected, returning');
        return;
    }

    // Build URL with filters
    const params = new URLSearchParams({
        page: state.page,
        page_size: state.promptsPageSize,
    });

    if (state.filters.minImages !== null) {
        params.append('min_images', state.filters.minImages);
    }
    if (state.filters.maxImages !== null) {
        params.append('max_images', state.filters.maxImages);
    }
    if (state.filters.promptSource) {
        params.append('prompt_source', state.filters.promptSource);
    }
    // Add model filter for prompts view
    if (state.promptsModelFilter && state.promptsModelFilter.length > 0) {
        params.append('models', state.promptsModelFilter.join(','));
    }

    // Use search API if there's a search query
    let url;
    if (state.searchQuery) {
        params.append('q', state.searchQuery);
        url = `api/subsets/${state.subset}/experiments/${state.experiment}/search/prompts?${params}`;
    } else {
        url = `api/subsets/${state.subset}/experiments/${state.experiment}/prompts?${params}`;
    }

    // Show loading state
    elements.promptsList.innerHTML = '<div class="loading">Loading prompts</div>';

    try {
        const data = await fetchJSON(url);

        state.totalPages = data.total_pages;
        state.totalBattles = data.total;

        renderPrompts(data.prompts);
        updatePagination();
        updateStats();

    } catch (error) {
        console.error('Failed to load prompts:', error);
        elements.promptsList.innerHTML = '<div class="empty-state"><p>Failed to load prompts</p></div>';
    }
}

async function loadBattleDetail(battle) {
    const battleId = `${battle.model_a}_vs_${battle.model_b}:${battle.sample_index}`;
    const url = `api/subsets/${state.subset}/experiments/${state.experiment}/battles/${battleId}`;

    try {
        const data = await fetchJSON(url);

        // Get input image count
        let inputImageCount = 1;
        try {
            const countData = await fetchJSON(`api/subsets/${state.subset}/samples/${battle.sample_index}/input_count`);
            inputImageCount = countData.count || 1;
        } catch (e) {
            // Ignore, default to 1
        }

        renderDetailModal(data, inputImageCount);
        showModal();
    } catch (error) {
        console.error('Failed to load battle detail:', error);
    }
}

async function updateStats() {
    if (!state.subset) return;

    try {
        const params = state.experiment ? `?exp_name=${state.experiment}` : '';
        const data = await fetchJSON(`api/subsets/${state.subset}/stats${params}`);

        const consistencyRate = (data.consistency_rate * 100).toFixed(1);

        elements.statsPanel.innerHTML = `
            <div class="stat-item">
                <span class="stat-label">Total Battles</span>
                <span class="stat-value">${data.total_battles}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Consistent</span>
                <span class="stat-value">${data.consistent_battles} (${consistencyRate}%)</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Ties</span>
                <span class="stat-value">${data.tie_battles}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Models</span>
                <span class="stat-value">${data.models.length}</span>
            </div>
        `;
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

// ========== ELO Leaderboard Functions ==========
async function loadEloLeaderboard() {
    if (!state.subset) {
        elements.eloPanel.innerHTML = '<p class="placeholder">Select a subset to view rankings</p>';
        return;
    }

    try {
        const data = await fetchJSON(`api/subsets/${state.subset}/leaderboard`);
        renderEloSidebar(data.leaderboard);
    } catch (error) {
        console.error('Failed to load ELO leaderboard:', error);
        elements.eloPanel.innerHTML = '<p class="placeholder">Failed to load rankings</p>';
    }
}

function renderEloSidebar(leaderboard) {
    if (!leaderboard || leaderboard.length === 0) {
        elements.eloPanel.innerHTML = '<p class="placeholder">No ELO data available</p>';
        return;
    }

    // Find min and max ELO for scaling the bars
    const elos = leaderboard.map(m => m.elo);
    const minElo = Math.min(...elos);
    const maxElo = Math.max(...elos);
    const eloRange = maxElo - minElo || 1;

    // Show all models in sidebar
    const displayList = leaderboard;

    elements.eloPanel.innerHTML = displayList.map(model => {
        const barWidth = ((model.elo - minElo) / eloRange * 70 + 30); // Min 30%, max 100%
        const rankClass = model.rank <= 3 ? `rank-${model.rank}` : '';

        return `
            <div class="elo-item" data-model="${escapeHtml(model.model)}" title="Click to view details">
                <span class="elo-rank ${rankClass}">#${model.rank}</span>
<span class="elo-model-name" title="${escapeHtml(model.model)}">${escapeHtml(truncateMiddle(getModelDisplayName(model.model), 10))}</span>
                <div class="elo-bar-container">
                    <div class="elo-bar" style="width: ${barWidth.toFixed(1)}%"></div>
                </div>
                <span class="elo-value">${Math.round(model.elo)}</span>
            </div>
        `;
    }).join('');

    // Add click handlers to show model details
    elements.eloPanel.querySelectorAll('.elo-item').forEach(item => {
        item.addEventListener('click', () => {
            const modelName = item.dataset.model;
            loadModelStats(modelName);
        });
    });
}

async function loadModelStats(modelName) {
    if (!state.subset) return;

    try {
        const data = await fetchJSON(`api/subsets/${state.subset}/models/${encodeURIComponent(modelName)}/stats`);
        renderModelStatsModal(data);
        showModelStatsModal();
    } catch (error) {
        console.error('Failed to load model stats:', error);
    }
}

function renderModelStatsModal(data) {
    const winRatePercent = (data.win_rate * 100).toFixed(1);

    let vsStatsHtml = '';
    if (data.vs_stats && data.vs_stats.length > 0) {
        vsStatsHtml = `
            <div class="vs-stats-section">
                <h3>Win Rate vs Opponents</h3>
                <table class="vs-stats-table">
                    <thead>
                        <tr>
                            <th>Opponent</th>
                            <th>W / L / T</th>
                            <th>Win Rate</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.vs_stats.map(vs => {
                            const vsWinRate = (vs.win_rate * 100).toFixed(1);
                            return `
                                <tr>
                                    <td class="opponent-cell">
                                        ${escapeHtml(getModelDisplayName(vs.opponent))}
                                        <span class="opponent-elo">(${Math.round(vs.opponent_elo)})</span>
                                    </td>
                                    <td class="wlt-cell">
                                        <span class="wins">${vs.wins}</span> /
                                        <span class="losses">${vs.losses}</span> /
                                        <span class="ties">${vs.ties}</span>
                                    </td>
                                    <td>
                                        <div class="win-rate-bar">
                                            <div class="win-rate-bar-bg">
                                                <div class="win-rate-bar-fill" style="width: ${vsWinRate}%"></div>
                                            </div>
                                            <span class="win-rate-text">${vsWinRate}%</span>
                                        </div>
                                    </td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    elements.modelStatsContent.innerHTML = `
        <div class="model-stats-header">
            <h2>${escapeHtml(getModelDisplayName(data.model))}</h2>
            <div class="model-stats-summary">
                <div class="model-stat-item">
                    <div class="stat-label">ELO Rating</div>
                    <div class="stat-value elo-value">${Math.round(data.elo)}</div>
                </div>
                <div class="model-stat-item">
                    <div class="stat-label">Wins</div>
                    <div class="stat-value wins-value">${data.wins}</div>
                </div>
                <div class="model-stat-item">
                    <div class="stat-label">Losses</div>
                    <div class="stat-value losses-value">${data.losses}</div>
                </div>
                <div class="model-stat-item">
                    <div class="stat-label">Ties</div>
                    <div class="stat-value ties-value">${data.ties}</div>
                </div>
                <div class="model-stat-item">
                    <div class="stat-label">Win Rate</div>
                    <div class="stat-value">${winRatePercent}%</div>
                </div>
                <div class="model-stat-item">
                    <div class="stat-label">Total Battles</div>
                    <div class="stat-value">${data.total_battles}</div>
                </div>
            </div>
        </div>
        ${vsStatsHtml}
    `;
}

function showModelStatsModal() {
    elements.modelStatsModal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

function hideModelStatsModal() {
    elements.modelStatsModal.classList.add('hidden');
    document.body.style.overflow = '';
}

async function loadFullLeaderboard() {
    if (!state.subset) return;

    try {
        const data = await fetchJSON(`api/subsets/${state.subset}/leaderboard`);
        renderFullLeaderboard(data.leaderboard);
        elements.leaderboardSubsetName.textContent = state.subset;
        showLeaderboardModal();
    } catch (error) {
        console.error('Failed to load full leaderboard:', error);
    }
}

function renderFullLeaderboard(leaderboard) {
    if (!leaderboard || leaderboard.length === 0) {
        elements.leaderboardContent.innerHTML = '<p class="placeholder">No ELO data available</p>';
        return;
    }

    elements.leaderboardContent.innerHTML = `
        <table class="leaderboard-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>ELO</th>
                    <th>Wins</th>
                    <th>Losses</th>
                    <th>Ties</th>
                    <th>Total</th>
                    <th class="win-rate-cell">Win Rate</th>
                </tr>
            </thead>
            <tbody>
                ${leaderboard.map(model => {
                    const rankClass = model.rank <= 3 ? `rank-${model.rank}` : '';
                    const winRatePercent = (model.win_rate * 100).toFixed(1);
                    return `
                        <tr data-model="${escapeHtml(model.model)}">
                            <td class="rank-cell ${rankClass}">#${model.rank}</td>
                            <td class="model-cell">${escapeHtml(getModelDisplayName(model.model))}</td>
                            <td class="elo-cell">${Math.round(model.elo)}</td>
                            <td class="stat-cell" style="color: var(--accent-green)">${model.wins}</td>
                            <td class="stat-cell" style="color: var(--accent-red)">${model.losses}</td>
                            <td class="stat-cell" style="color: var(--accent-yellow)">${model.ties}</td>
                            <td class="stat-cell">${model.total_battles}</td>
                            <td class="win-rate-cell">
                                <div class="win-rate-bar">
                                    <div class="win-rate-bar-bg">
                                        <div class="win-rate-bar-fill" style="width: ${winRatePercent}%"></div>
                                    </div>
                                    <span class="win-rate-text">${winRatePercent}%</span>
                                </div>
                            </td>
                        </tr>
                    `;
                }).join('')}
            </tbody>
        </table>
    `;

    // Add click handlers to show model details
    elements.leaderboardContent.querySelectorAll('tbody tr').forEach(row => {
        row.addEventListener('click', () => {
            const modelName = row.dataset.model;
            hideLeaderboardModal();
            loadModelStats(modelName);
        });
    });
}

function showLeaderboardModal() {
    elements.leaderboardModal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

function hideLeaderboardModal() {
    elements.leaderboardModal.classList.add('hidden');
    document.body.style.overflow = '';
}

// ========== Render Functions ==========
function renderBattles(battles) {
    if (!battles || battles.length === 0) {
        elements.battleList.innerHTML = '<div class="empty-state"><p>No battles found</p></div>';
        return;
    }

    elements.battleList.innerHTML = battles.map(battle => renderBattleCard(battle)).join('');

    // Add click handlers for battle detail
    elements.battleList.querySelectorAll('.battle-card').forEach((card, index) => {
        // Don't open detail when clicking favorite button
        card.addEventListener('click', (e) => {
            if (!e.target.closest('.btn-favorite-toggle')) {
                loadBattleDetail(battles[index]);
            }
        });
    });

    // Add click handlers for favorite buttons
    elements.battleList.querySelectorAll('.btn-favorite-toggle').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const sampleIndex = parseInt(btn.dataset.sampleIndex);
            const instruction = btn.dataset.instruction || '';
            const added = toggleFavorite(state.subset, state.experiment, sampleIndex, instruction);

            // Update button appearance
            btn.classList.toggle('favorited', added);
            btn.textContent = added ? '★' : '☆';
            btn.title = added ? 'Remove from favorites' : 'Add to favorites';
        });
    });
}

function renderPrompts(prompts) {
    if (!prompts || prompts.length === 0) {
        elements.promptsList.innerHTML = '<div class="empty-state"><p>No prompts found</p></div>';
        return;
    }

    elements.promptsList.innerHTML = prompts.map(prompt => renderPromptCard(prompt)).join('');

    // Add click handlers for images (lightbox)
    elements.promptsList.querySelectorAll('.prompt-input-image img, .prompt-model-image img').forEach(img => {
        img.addEventListener('click', (e) => {
            e.stopPropagation();
            openLightbox(img.src, img.alt || img.dataset.label || '');
        });
    });

    // Add click handlers for favorite buttons
    elements.promptsList.querySelectorAll('.btn-favorite-toggle').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const sampleIndex = parseInt(btn.dataset.sampleIndex);
            const instruction = btn.dataset.instruction || '';
            const added = toggleFavorite(state.subset, state.experiment, sampleIndex, instruction);

            // Update button appearance
            btn.classList.toggle('favorited', added);
            btn.textContent = added ? '★' : '☆';
            btn.title = added ? 'Remove from favorites' : 'Add to favorites';
        });
    });

    // Add click handlers for model names to show battle details
    elements.promptsList.querySelectorAll('.prompt-model-name.clickable').forEach(nameEl => {
        nameEl.addEventListener('click', (e) => {
            e.stopPropagation();
            const model = nameEl.dataset.model;
            const sampleIndex = parseInt(nameEl.dataset.sampleIndex);
            const subset = nameEl.dataset.subset;
            showModelBattlesModal(subset, state.experiment, sampleIndex, model);
        });
    });
}

function renderPromptCard(prompt) {
    const inputImagesHtml = renderPromptInputImages(prompt.subset, prompt.sample_index, prompt.input_image_count || 1);
    const modelsHtml = renderPromptModelsGrid(prompt.subset, prompt.sample_index, prompt.models || []);
    const favorited = isFavorited(state.subset, state.experiment, prompt.sample_index);

    return `
        <div class="prompt-card">
            <div class="prompt-card-header">
                <div class="prompt-card-info">
                    <div class="prompt-card-instruction">${escapeHtml(prompt.instruction || 'No instruction')}</div>
                    <div class="prompt-card-meta">
                        <span>Index: ${prompt.sample_index}</span>
                        ${state.experiment === '__all__' ? `<span>Exp: ${escapeHtml(prompt.exp_name)}</span>` : ''}
                        ${prompt.task_type ? `<span>Task: ${escapeHtml(prompt.task_type)}</span>` : ''}
                        ${prompt.prompt_source ? `<span>Source: ${escapeHtml(prompt.prompt_source)}</span>` : ''}
                        <span>Input Images: ${prompt.input_image_count || 1}</span>
                    </div>
                </div>
                <div class="prompt-card-actions">
                    <button class="btn-favorite-toggle ${favorited ? 'favorited' : ''}"
                            data-sample-index="${prompt.sample_index}"
                            data-instruction="${escapeHtml(prompt.instruction || '')}"
                            title="${favorited ? 'Remove from favorites' : 'Add to favorites'}">
                        ${favorited ? '★' : '☆'}
                    </button>
                </div>
            </div>
            ${inputImagesHtml}
            ${modelsHtml}
        </div>
    `;
}

function renderPromptInputImages(subset, sampleIndex, count) {
    if (count === 0) return '';

    let imagesHtml = '';
    for (let i = 0; i < count; i++) {
        const imgUrl = `images/${subset}/input/${sampleIndex}/${i}`;
        imagesHtml += `
            <div class="prompt-input-image">
                <img src="${imgUrl}" alt="Input ${i + 1}" loading="lazy">
            </div>
        `;
    }

    return `
        <div class="prompt-input-section">
            <div class="prompt-input-title">Input Images (${count})</div>
            <div class="prompt-input-images">
                ${imagesHtml}
            </div>
        </div>
    `;
}

function renderPromptModelsGrid(subset, sampleIndex, models) {
    if (models.length === 0) {
        return `
            <div class="prompt-models-section">
                <div class="prompt-models-title">Model Outputs</div>
                <p class="placeholder">No model outputs available</p>
            </div>
        `;
    }

    const modelsHtml = models.map((m, idx) => {
        const rank = idx + 1;
        const imgUrl = `images/${subset}/${encodeURIComponent(m.model)}/${sampleIndex}`;
        const winRatePercent = (m.win_rate * 100).toFixed(1);

        let rankClass = '';
        let rankBadge = '';
        if (rank === 1) {
            rankClass = 'rank-1';
            rankBadge = '<span class="prompt-model-rank rank-1">🥇</span>';
        } else if (rank === 2) {
            rankClass = 'rank-2';
            rankBadge = '<span class="prompt-model-rank rank-2">🥈</span>';
        } else if (rank === 3) {
            rankClass = 'rank-3';
            rankBadge = '<span class="prompt-model-rank rank-3">🥉</span>';
        }

        return `
            <div class="prompt-model-card ${rankClass}" data-model="${escapeHtml(m.model)}" data-sample-index="${sampleIndex}" data-subset="${subset}">
                <div class="prompt-model-image">
                    <img src="${imgUrl}" alt="${escapeHtml(m.model)}" data-label="${escapeHtml(getModelDisplayName(m.model))}" loading="lazy">
                </div>
                <div class="prompt-model-info">
                    <div class="prompt-model-name clickable" data-model="${escapeHtml(m.model)}" data-sample-index="${sampleIndex}" data-subset="${subset}">${rankBadge}${escapeHtml(getModelDisplayName(m.model))}</div>
                    <div class="prompt-model-stats">
                        <span class="win-rate">${winRatePercent}%</span>
                        (<span class="wins">${m.wins}W</span>/<span class="losses">${m.losses}L</span>/<span class="ties">${m.ties}T</span>)
                    </div>
                </div>
            </div>
        `;
    }).join('');

    return `
        <div class="prompt-models-section">
            <div class="prompt-models-title">Model Outputs (sorted by win rate)</div>
            <div class="prompt-models-grid">
                ${modelsHtml}
            </div>
        </div>
    `;
}

function renderBattleCard(battle) {
    const selectedModels = state.filters.models || [];
    const isSingleModelFilter = selectedModels.length === 1;
    const selectedModel = isSingleModelFilter ? selectedModels[0] : null;

    const isWin = selectedModel && battle.final_winner === selectedModel;
    const isLoss = selectedModel && battle.final_winner !== 'tie' && battle.final_winner !== selectedModel;
    const isTie = battle.final_winner === 'tie';

    // Determine winner/loser styling for model names
    let modelAClass = 'model-name';
    let modelBClass = 'model-name';

    if (battle.final_winner === battle.model_a) {
        modelAClass += ' winner';
        modelBClass += ' loser';
    } else if (battle.final_winner === battle.model_b) {
        modelBClass += ' winner';
        modelAClass += ' loser';
    }

    // Result badge
    let resultBadge = '';
    if (isTie) {
        resultBadge = '<span class="badge badge-tie">Tie</span>';
    } else if (selectedModel) {
        if (isWin) {
            resultBadge = '<span class="badge badge-win">Win</span>';
        } else if (isLoss) {
            resultBadge = '<span class="badge badge-loss">Loss</span>';
        }
    }

    // Consistency badge
    const consistencyBadge = battle.is_consistent
        ? '<span class="badge badge-consistent">Consistent</span>'
        : '<span class="badge badge-inconsistent">Inconsistent</span>';

    // Favorite button
    const favorited = isFavorited(state.subset, state.experiment, battle.sample_index);
    const favoriteBtn = `
        <button class="btn-favorite-toggle ${favorited ? 'favorited' : ''}"
                data-sample-index="${battle.sample_index}"
                data-instruction="${escapeHtml(battle.instruction || '')}"
                title="${favorited ? 'Remove from favorites' : 'Add to favorites'}">
            ${favorited ? '★' : '☆'}
        </button>
    `;

    // Generate input images HTML (support multiple)
    const inputImageCount = battle.input_image_count || 1;
    let inputImagesHtml = '';

    if (inputImageCount === 1) {
        // Single input image - normal size
        const inputImageUrl = `images/${state.subset}/input/${battle.sample_index}/0`;
        inputImagesHtml = `
            <div class="battle-image-container">
                <img src="${inputImageUrl}" alt="Input" loading="lazy">
                <span class="image-label">Input</span>
            </div>
        `;
    } else {
        // Multiple input images - show in a grid within the input column
        let inputThumbs = '';
        for (let i = 0; i < inputImageCount; i++) {
            const inputImageUrl = `images/${state.subset}/input/${battle.sample_index}/${i}`;
            inputThumbs += `
                <div class="input-thumb">
                    <img src="${inputImageUrl}" alt="Input ${i + 1}" loading="lazy">
                </div>
            `;
        }
        inputImagesHtml = `
            <div class="battle-image-container multi-input" data-count="${inputImageCount}">
                <div class="input-thumbs-grid">${inputThumbs}</div>
                <span class="image-label">Input (${inputImageCount})</span>
            </div>
        `;
    }

    const modelAImageUrl = `images/${state.subset}/${encodeURIComponent(battle.model_a)}/${battle.sample_index}`;
    const modelBImageUrl = `images/${state.subset}/${encodeURIComponent(battle.model_b)}/${battle.sample_index}`;

    return `
        <div class="battle-card" data-id="${battle.id}">
            <div class="battle-card-header">
                <div class="battle-models">
                    <span class="${modelAClass}">${escapeHtml(getModelDisplayName(battle.model_a))}</span>
                    <span class="vs-label">vs</span>
                    <span class="${modelBClass}">${escapeHtml(getModelDisplayName(battle.model_b))}</span>
                </div>
                <div class="battle-badges">
                    ${favoriteBtn}
                    ${resultBadge}
                    ${consistencyBadge}
                </div>
            </div>
            <div class="battle-instruction">${escapeHtml(battle.instruction || 'No instruction')}</div>
            <div class="battle-images">
                ${inputImagesHtml}
                <div class="battle-image-container">
                    <img src="${modelAImageUrl}" alt="${escapeHtml(battle.model_a)}" loading="lazy">
                    <span class="image-label">${escapeHtml(getModelDisplayName(battle.model_a))}</span>
                </div>
                <div class="battle-image-container">
                    <img src="${modelBImageUrl}" alt="${escapeHtml(battle.model_b)}" loading="lazy">
                    <span class="image-label">${escapeHtml(getModelDisplayName(battle.model_b))}</span>
                </div>
            </div>
            <div class="battle-meta">
                <span>Index: ${battle.sample_index}</span>
                ${state.experiment === '__all__' ? `<span>Exp: ${escapeHtml(battle.exp_name)}</span>` : ''}
                ${battle.prompt_source ? `<span>Source: ${escapeHtml(battle.prompt_source)}</span>` : ''}
                <span>Winner: ${escapeHtml(battle.winner_display)}</span>
            </div>
        </div>
    `;
}

function renderDetailModal(battle, inputImageCount = 1) {
    // Determine winner/loser classes
    const modelAIsWinner = battle.final_winner === battle.model_a;
    const modelBIsWinner = battle.final_winner === battle.model_b;
    const modelAClass = modelAIsWinner ? 'winner' : (battle.final_winner !== 'tie' ? 'loser' : '');
    const modelBClass = modelBIsWinner ? 'winner' : (battle.final_winner !== 'tie' ? 'loser' : '');

    // Generate input image elements
    let inputImagesHtml = '';
    for (let i = 0; i < inputImageCount; i++) {
        const inputImageUrl = `images/${state.subset}/input/${battle.sample_index}/${i}`;
        const inputLabel = inputImageCount > 1 ? `Input ${i + 1}` : 'Input';
        inputImagesHtml += `
            <div class="detail-image-box input-image">
                <h4>${inputLabel}</h4>
                <img src="${inputImageUrl}" alt="${inputLabel}" data-label="${inputLabel}" class="zoomable">
            </div>
        `;
    }

    const modelAImageUrl = `images/${state.subset}/${encodeURIComponent(battle.model_a)}/${battle.sample_index}`;
    const modelBImageUrl = `images/${state.subset}/${encodeURIComponent(battle.model_b)}/${battle.sample_index}`;

    // VLM outputs
    let vlmOutputsHtml = '';

    if (battle.original_call || battle.swapped_call) {
        vlmOutputsHtml = '<div class="detail-vlm-outputs"><h3>VLM Judge Outputs</h3>';

        if (battle.original_call) {
            const parsed = battle.original_call.parsed_result || {};
            vlmOutputsHtml += `
                <div class="vlm-call">
                    <h4>Original Order (A=${escapeHtml(getModelDisplayName(battle.model_a))}, B=${escapeHtml(getModelDisplayName(battle.model_b))})</h4>
                    <div class="vlm-call-meta">
                        Winner: <strong>${escapeHtml(parsed.winner || 'N/A')}</strong> |
                        Parse: ${battle.original_call.parse_success ? '✓' : '✗'}
                    </div>
                    <div class="vlm-response">${escapeHtml(battle.original_call.raw_response || 'No response')}</div>
                </div>
            `;
        }

        if (battle.swapped_call) {
            const parsed = battle.swapped_call.parsed_result || {};
            vlmOutputsHtml += `
                <div class="vlm-call">
                    <h4>Swapped Order (A=${escapeHtml(getModelDisplayName(battle.model_b))}, B=${escapeHtml(getModelDisplayName(battle.model_a))})</h4>
                    <div class="vlm-call-meta">
                        Winner: <strong>${escapeHtml(parsed.winner || 'N/A')}</strong> |
                        Parse: ${battle.swapped_call.parse_success ? '✓' : '✗'}
                    </div>
                    <div class="vlm-response">${escapeHtml(battle.swapped_call.raw_response || 'No response')}</div>
                </div>
            `;
        }

        vlmOutputsHtml += '</div>';
    } else {
        vlmOutputsHtml = `
            <div class="detail-vlm-outputs">
                <h3>VLM Judge Outputs</h3>
                <p class="placeholder">Audit logs not available for this battle</p>
            </div>
        `;
    }

    // Format original_metadata as JSON if it exists
    let originalMetadataHtml = '';
    if (battle.original_metadata) {
        try {
            const metaStr = typeof battle.original_metadata === 'string'
                ? battle.original_metadata
                : JSON.stringify(battle.original_metadata, null, 2);
            originalMetadataHtml = `
                <div class="detail-metadata-section">
                    <h4>Original Metadata</h4>
                    <pre class="metadata-json">${escapeHtml(metaStr)}</pre>
                </div>
            `;
        } catch (e) {
            originalMetadataHtml = `
                <div class="detail-metadata-section">
                    <h4>Original Metadata</h4>
                    <pre class="metadata-json">${escapeHtml(String(battle.original_metadata))}</pre>
                </div>
            `;
        }
    }

    elements.modalContent.innerHTML = `
        <div class="detail-header">
            <h2>
                <span class="model-name ${battle.final_winner === battle.model_a ? 'winner' : (battle.final_winner !== 'tie' ? 'loser' : '')}">${escapeHtml(getModelDisplayName(battle.model_a))}</span>
                <span class="vs-label">vs</span>
                <span class="model-name ${battle.final_winner === battle.model_b ? 'winner' : (battle.final_winner !== 'tie' ? 'loser' : '')}">${escapeHtml(getModelDisplayName(battle.model_b))}</span>
                <span class="badge ${battle.is_consistent ? 'badge-consistent' : 'badge-inconsistent'}">${battle.is_consistent ? 'Consistent' : 'Inconsistent'}</span>
            </h2>
            <div class="detail-meta-info">
                <span class="meta-tag"><strong>Index:</strong> ${battle.sample_index}</span>
                ${battle.task_type ? `<span class="meta-tag"><strong>Task:</strong> ${escapeHtml(battle.task_type)}</span>` : ''}
                ${battle.prompt_source ? `<span class="meta-tag"><strong>Source:</strong> ${escapeHtml(battle.prompt_source)}</span>` : ''}
            </div>
            <div class="detail-instruction">${escapeHtml(battle.instruction || 'No instruction')}</div>
            ${originalMetadataHtml}
        </div>

        <div class="detail-images">
            ${inputImagesHtml}
            <div class="detail-image-box output-image ${modelAClass}">
                <h4>${escapeHtml(getModelDisplayName(battle.model_a))} ${modelAIsWinner ? '👑' : ''}</h4>
                <img src="${modelAImageUrl}" alt="${escapeHtml(battle.model_a)}" data-label="${escapeHtml(getModelDisplayName(battle.model_a))}" class="zoomable">
            </div>
            <div class="detail-image-box output-image ${modelBClass}">
                <h4>${escapeHtml(getModelDisplayName(battle.model_b))} ${modelBIsWinner ? '👑' : ''}</h4>
                <img src="${modelBImageUrl}" alt="${escapeHtml(battle.model_b)}" data-label="${escapeHtml(getModelDisplayName(battle.model_b))}" class="zoomable">
            </div>
        </div>

        ${vlmOutputsHtml}
    `;

    // Add click handlers for zoomable images
    setTimeout(() => {
        elements.modalContent.querySelectorAll('img.zoomable').forEach(img => {
            img.addEventListener('click', (e) => {
                e.stopPropagation();
                openLightbox(img.src, img.dataset.label || img.alt);
            });
        });
    }, 0);
}

function updatePagination() {
    const start = (state.page - 1) * state.pageSize + 1;
    const end = Math.min(state.page * state.pageSize, state.totalBattles);

    elements.paginationInfo.textContent = state.totalBattles > 0
        ? `Showing ${start}-${end} of ${state.totalBattles}`
        : '';

    const canPrev = state.page > 1;
    const canNext = state.page < state.totalPages;

    elements.firstPage.disabled = !canPrev;
    elements.prevPage.disabled = !canPrev;
    elements.nextPage.disabled = !canNext;
    elements.lastPage.disabled = !canNext;
    elements.firstPageBottom.disabled = !canPrev;
    elements.prevPageBottom.disabled = !canPrev;
    elements.nextPageBottom.disabled = !canNext;
    elements.lastPageBottom.disabled = !canNext;

    // Update page input max value and placeholder
    elements.pageInput.max = state.totalPages;
    elements.pageInput.placeholder = `1-${state.totalPages}`;
    elements.pageInputBottom.max = state.totalPages;
    elements.pageInputBottom.placeholder = `1-${state.totalPages}`;

    // Render page numbers
    renderPageNumbers(elements.pageNumbers);
    renderPageNumbers(elements.pageNumbersBottom);
}

function renderPageNumbers(container) {
    const total = state.totalPages || 1;
    const current = state.page;

    // Generate page numbers with ellipsis
    const pages = [];
    const maxVisible = 7; // Max visible page numbers

    if (total <= maxVisible) {
        // Show all pages
        for (let i = 1; i <= total; i++) {
            pages.push(i);
        }
    } else {
        // Always show first page
        pages.push(1);

        if (current > 3) {
            pages.push('...');
        }

        // Pages around current
        const start = Math.max(2, current - 1);
        const end = Math.min(total - 1, current + 1);

        for (let i = start; i <= end; i++) {
            if (!pages.includes(i)) {
                pages.push(i);
            }
        }

        if (current < total - 2) {
            pages.push('...');
        }

        // Always show last page
        if (!pages.includes(total)) {
            pages.push(total);
        }
    }

    container.innerHTML = pages.map(p => {
        if (p === '...') {
            return '<span class="page-number ellipsis">...</span>';
        }
        const activeClass = p === current ? 'active' : '';
        return `<button class="page-number ${activeClass}" data-page="${p}">${p}</button>`;
    }).join('');

    // Add click handlers
    container.querySelectorAll('.page-number:not(.ellipsis)').forEach(btn => {
        btn.addEventListener('click', () => {
            const page = parseInt(btn.dataset.page);
            if (page !== state.page) {
                state.page = page;
                loadCurrentView();
            }
        });
    });
}

// ========== Modal Functions ==========
function showModal() {
    elements.modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

function hideModal() {
    elements.modal.classList.add('hidden');
    document.body.style.overflow = '';
}

// ========== Model Battles Modal Functions ==========
// State for model battles modal
let modelBattlesState = {
    subset: null,
    expName: null,
    sampleIndex: null,
    model: null,
    allOpponents: [],
    selectedOpponents: new Set(),
    battles: [],
};

async function showModelBattlesModal(subset, expName, sampleIndex, model) {
    // Store current state
    modelBattlesState.subset = subset;
    modelBattlesState.expName = expName;
    modelBattlesState.sampleIndex = sampleIndex;
    modelBattlesState.model = model;

    // Show loading state
    elements.modalContent.innerHTML = `
        <div class="model-battles-modal">
            <div class="model-battles-header">
                <h2>Battle Records: ${escapeHtml(getModelDisplayName(model))}</h2>
                <p class="model-battles-subtitle">Sample Index: ${sampleIndex}</p>
            </div>
            <div class="loading">Loading battle records...</div>
        </div>
    `;
    showModal();

    try {
        // Fetch battle data
        const url = `api/subsets/${subset}/experiments/${expName}/samples/${sampleIndex}/models/${encodeURIComponent(model)}/battles`;
        const data = await fetchJSON(url);

        // Store data
        modelBattlesState.allOpponents = data.all_opponents || [];
        modelBattlesState.selectedOpponents = new Set(modelBattlesState.allOpponents);
        modelBattlesState.battles = data.battles || [];

        // Render full modal content
        renderModelBattlesModal(data);
    } catch (error) {
        console.error('Failed to load model battles:', error);
        elements.modalContent.innerHTML = `
            <div class="model-battles-modal">
                <div class="model-battles-header">
                    <h2>Battle Records: ${escapeHtml(getModelDisplayName(model))}</h2>
                </div>
                <div class="empty-state"><p>Failed to load battle records</p></div>
            </div>
        `;
    }
}

function renderModelBattlesModal(data) {
    const { model, sample_index, wins, losses, ties, total, win_rate, battles, all_opponents } = data;
    const winRatePercent = (win_rate * 100).toFixed(1);

    // Filter battles based on selected opponents
    const filteredBattles = battles.filter(b => modelBattlesState.selectedOpponents.has(b.opponent));

    // Group battles by opponent
    const battlesByOpponent = {};
    filteredBattles.forEach(b => {
        if (!battlesByOpponent[b.opponent]) {
            battlesByOpponent[b.opponent] = [];
        }
        battlesByOpponent[b.opponent].push(b);
    });

    // Calculate stats per opponent
    const opponentStats = {};
    all_opponents.forEach(opponent => {
        const opponentBattles = battles.filter(b => b.opponent === opponent);
        let w = 0, l = 0, t = 0;
        opponentBattles.forEach(b => {
            if (b.result === 'win') w++;
            else if (b.result === 'loss') l++;
            else t++;
        });
        const opTotal = w + l + t;
        opponentStats[opponent] = {
            wins: w,
            losses: l,
            ties: t,
            total: opTotal,
            winRate: opTotal > 0 ? ((w / opTotal) * 100).toFixed(1) : '0.0',
        };
    });

    // Helper function to render judge call
    const renderJudgeCall = (call, label, modelA, modelB) => {
        if (!call) return '';
        const parsed = call.parsed_result || {};
        const winner = parsed.winner || 'N/A';
        const parseSuccess = call.parse_success ? '✓' : '✗';
        const rawResponse = call.raw_response || 'No response';

        return `
            <div class="judge-call">
                <div class="judge-call-header">
                    <span class="judge-call-label">${label}</span>
                    <span class="judge-call-order">(A=${escapeHtml(getModelDisplayName(modelA))}, B=${escapeHtml(getModelDisplayName(modelB))})</span>
                </div>
                <div class="judge-call-meta">
                    Winner: <strong>${escapeHtml(winner)}</strong> | Parse: ${parseSuccess}
                </div>
                <div class="judge-call-response">${escapeHtml(rawResponse)}</div>
            </div>
        `;
    };

    // Generate opponent sections (collapsible)
    const sortedOpponents = Object.keys(battlesByOpponent).sort();
    const opponentSectionsHtml = sortedOpponents.length > 0 ? sortedOpponents.map(opponent => {
        const opponentBattles = battlesByOpponent[opponent];
        const stats = opponentStats[opponent];
        const isSelected = modelBattlesState.selectedOpponents.has(opponent);

        // Determine overall result against this opponent
        let overallResultClass = '';
        if (stats.wins > stats.losses) {
            overallResultClass = 'result-win';
        } else if (stats.losses > stats.wins) {
            overallResultClass = 'result-loss';
        } else if (stats.ties > 0 && stats.wins === 0 && stats.losses === 0) {
            overallResultClass = 'result-tie';
        }

        // Generate battle records for this opponent
        const battlesHtml = opponentBattles.map((b, idx) => {
            const resultClass = b.result === 'win' ? 'result-win' : (b.result === 'loss' ? 'result-loss' : 'result-tie');
            const resultText = b.result === 'win' ? 'Win' : (b.result === 'loss' ? 'Loss' : 'Tie');
            const consistentBadge = b.is_consistent
                ? '<span class="badge badge-consistent">Consistent</span>'
                : '<span class="badge badge-inconsistent">Inconsistent</span>';

            // Render judge outputs
            const hasJudgeOutputs = b.original_call || b.swapped_call;
            let judgeOutputsHtml = '';
            if (hasJudgeOutputs) {
                judgeOutputsHtml = `
                    <div class="battle-judge-outputs">
                        ${renderJudgeCall(b.original_call, 'Original Order', b.model_a, b.model_b)}
                        ${renderJudgeCall(b.swapped_call, 'Swapped Order', b.model_b, b.model_a)}
                    </div>
                `;
            } else {
                judgeOutputsHtml = `
                    <div class="battle-judge-outputs">
                        <p class="placeholder">Judge outputs not available</p>
                    </div>
                `;
            }

            return `
                <div class="battle-record-item">
                    <div class="battle-record-item-header">
                        <span class="badge ${resultClass}">${resultText}</span>
                        ${consistentBadge}
                        ${b.exp_name ? `<span class="battle-exp-name">${escapeHtml(b.exp_name)}</span>` : ''}
                    </div>
                    ${judgeOutputsHtml}
                </div>
            `;
        }).join('');

        return `
            <div class="opponent-section ${isSelected ? '' : 'hidden'}" data-opponent="${escapeHtml(opponent)}">
                <div class="opponent-section-header" onclick="this.parentElement.classList.toggle('expanded')">
                    <div class="opponent-section-info">
                        <span class="opponent-name ${overallResultClass}">vs ${escapeHtml(getModelDisplayName(opponent))}</span>
                        <span class="opponent-stats">
                            ${stats.winRate}% (<span class="wins">${stats.wins}W</span>/<span class="losses">${stats.losses}L</span>/<span class="ties">${stats.ties}T</span>)
                        </span>
                    </div>
                    <span class="expand-icon">▼</span>
                </div>
                <div class="opponent-section-content">
                    ${battlesHtml}
                </div>
            </div>
        `;
    }).join('') : '<p class="empty-state">No battles match the current filter</p>';

    // Generate opponent filter checkboxes
    const opponentCheckboxesHtml = all_opponents.map(opponent => {
        const checked = modelBattlesState.selectedOpponents.has(opponent) ? 'checked' : '';
        return `
            <label class="opponent-checkbox">
                <input type="checkbox" value="${escapeHtml(opponent)}" ${checked}>
                <span>${escapeHtml(getModelDisplayName(opponent))}</span>
            </label>
        `;
    }).join('');

    // Calculate filtered stats
    let filteredWins = 0, filteredLosses = 0, filteredTies = 0;
    filteredBattles.forEach(b => {
        if (b.result === 'win') filteredWins++;
        else if (b.result === 'loss') filteredLosses++;
        else filteredTies++;
    });
    const filteredTotal = filteredWins + filteredLosses + filteredTies;
    const filteredWinRate = filteredTotal > 0 ? ((filteredWins / filteredTotal) * 100).toFixed(1) : '0.0';

    elements.modalContent.innerHTML = `
        <div class="model-battles-modal">
            <div class="model-battles-header">
                <h2>Battle Records: ${escapeHtml(getModelDisplayName(model))}</h2>
                <p class="model-battles-subtitle">Sample Index: ${sample_index}</p>
                <div class="model-battles-stats">
                    <span class="stat-item"><strong>Overall:</strong> ${winRatePercent}% win rate (${wins}W / ${losses}L / ${ties}T)</span>
                    <span class="stat-item"><strong>Filtered:</strong> ${filteredWinRate}% win rate (${filteredWins}W / ${filteredLosses}L / ${filteredTies}T)</span>
                </div>
            </div>

            <div class="model-battles-filter">
                <div class="filter-header">
                    <h4>Filter by Opponent:</h4>
                    <div class="filter-actions">
                        <button class="btn btn-small" id="select-all-opponents">Select All</button>
                        <button class="btn btn-small" id="clear-all-opponents">Clear All</button>
                    </div>
                </div>
                <div class="opponent-checkboxes">
                    ${opponentCheckboxesHtml}
                </div>
            </div>

            <div class="model-battles-list">
                <h4>Battle Records by Opponent (${sortedOpponents.length} opponents, ${filteredBattles.length} battles)</h4>
                <p class="model-battles-hint">Click on an opponent to expand/collapse battle details</p>
                <div class="opponent-sections-container">
                    ${opponentSectionsHtml}
                </div>
            </div>
        </div>
    `;

    // Add event listeners for filter checkboxes
    setTimeout(() => {
        // Opponent checkbox change
        elements.modalContent.querySelectorAll('.opponent-checkbox input').forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                if (checkbox.checked) {
                    modelBattlesState.selectedOpponents.add(checkbox.value);
                } else {
                    modelBattlesState.selectedOpponents.delete(checkbox.value);
                }
                // Re-render with current data
                renderModelBattlesModal({
                    model: modelBattlesState.model,
                    sample_index: modelBattlesState.sampleIndex,
                    wins, losses, ties, total, win_rate,
                    battles: modelBattlesState.battles,
                    all_opponents: modelBattlesState.allOpponents,
                });
            });
        });

        // Select all button
        const selectAllBtn = elements.modalContent.querySelector('#select-all-opponents');
        if (selectAllBtn) {
            selectAllBtn.addEventListener('click', () => {
                modelBattlesState.selectedOpponents = new Set(modelBattlesState.allOpponents);
                renderModelBattlesModal({
                    model: modelBattlesState.model,
                    sample_index: modelBattlesState.sampleIndex,
                    wins, losses, ties, total, win_rate,
                    battles: modelBattlesState.battles,
                    all_opponents: modelBattlesState.allOpponents,
                });
            });
        }

        // Clear all button
        const clearAllBtn = elements.modalContent.querySelector('#clear-all-opponents');
        if (clearAllBtn) {
            clearAllBtn.addEventListener('click', () => {
                modelBattlesState.selectedOpponents = new Set();
                renderModelBattlesModal({
                    model: modelBattlesState.model,
                    sample_index: modelBattlesState.sampleIndex,
                    wins, losses, ties, total, win_rate,
                    battles: modelBattlesState.battles,
                    all_opponents: modelBattlesState.allOpponents,
                });
            });
        }
    }, 0);
}

// ========== Lightbox Functions ==========
function openLightbox(src, label) {
    elements.lightboxImg.src = src;
    elements.lightboxLabel.textContent = label || '';
    elements.lightbox.classList.add('active');
}

function closeLightbox() {
    elements.lightbox.classList.remove('active');
    elements.lightboxImg.src = '';
}

// ========== Favorites Functions ==========
function loadFavoritesFromStorage() {
    try {
        const stored = localStorage.getItem('genarena_favorites');
        if (stored) {
            state.favorites = JSON.parse(stored);
        }
    } catch (e) {
        console.error('Failed to load favorites:', e);
        state.favorites = [];
    }
    updateFavoritesCount();
}

function saveFavoritesToStorage() {
    try {
        localStorage.setItem('genarena_favorites', JSON.stringify(state.favorites));
    } catch (e) {
        console.error('Failed to save favorites:', e);
    }
    updateFavoritesCount();
}

function updateFavoritesCount() {
    elements.favoritesCount.textContent = state.favorites.length;
}

function isFavorited(subset, expName, sampleIndex) {
    return state.favorites.some(
        f => f.subset === subset && f.exp_name === expName && f.sample_index === sampleIndex
    );
}

function toggleFavorite(subset, expName, sampleIndex, instruction = '') {
    const index = state.favorites.findIndex(
        f => f.subset === subset && f.exp_name === expName && f.sample_index === sampleIndex
    );

    if (index >= 0) {
        // Remove from favorites
        state.favorites.splice(index, 1);
    } else {
        // Add to favorites
        state.favorites.push({
            subset,
            exp_name: expName,
            sample_index: sampleIndex,
            instruction: instruction,
            added_at: new Date().toISOString()
        });
    }

    saveFavoritesToStorage();
    return index < 0; // Returns true if added, false if removed
}

function removeFavorite(subset, expName, sampleIndex) {
    const index = state.favorites.findIndex(
        f => f.subset === subset && f.exp_name === expName && f.sample_index === sampleIndex
    );

    if (index >= 0) {
        state.favorites.splice(index, 1);
        saveFavoritesToStorage();
    }
}

function clearAllFavorites() {
    if (confirm('Are you sure you want to clear all favorites?')) {
        state.favorites = [];
        saveFavoritesToStorage();
        renderFavoritesModal();
    }
}

function showFavoritesModal() {
    elements.favoritesModal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
    renderFavoritesModal();
}

function hideFavoritesModal() {
    elements.favoritesModal.classList.add('hidden');
    document.body.style.overflow = '';
}

async function renderFavoritesModal() {
    // First, render the model filter checkboxes
    renderFavoritesModelFilter();

    if (state.favorites.length === 0) {
        elements.favoritesContent.innerHTML = `
            <div class="favorites-empty">
                <p>No favorite prompts yet.</p>
                <p>Click the ☆ icon on any battle card to add it to favorites.</p>
            </div>
        `;
        return;
    }

    elements.favoritesContent.innerHTML = '<div class="favorite-loading">Loading favorites...</div>';

    // Build query params for model filter and stats scope
    const params = [];
    if (state.favoritesModelFilter.length > 0) {
        params.push(`models=${state.favoritesModelFilter.join(',')}`);
    }
    params.push(`stats_scope=${state.favoritesStatsScope}`);
    const queryString = params.length > 0 ? `?${params.join('&')}` : '';

    // Load all favorites data
    const favoritesHtml = [];

    for (const fav of state.favorites) {
        try {
            const url = `api/subsets/${fav.subset}/experiments/${fav.exp_name}/samples/${fav.sample_index}/all_models${queryString}`;
            const data = await fetchJSON(url);
            favoritesHtml.push(renderFavoritePromptCard(fav, data));
        } catch (e) {
            console.error('Failed to load favorite:', fav, e);
            favoritesHtml.push(renderFavoritePromptCardError(fav));
        }
    }

    elements.favoritesContent.innerHTML = favoritesHtml.join('');

    // Add event handlers
    elements.favoritesContent.querySelectorAll('.btn-unfavorite').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const subset = btn.dataset.subset;
            const expName = btn.dataset.expName;
            const sampleIndex = parseInt(btn.dataset.sampleIndex);
            removeFavorite(subset, expName, sampleIndex);
            renderFavoritesModal();
        });
    });

    // Add image click handlers for lightbox
    elements.favoritesContent.querySelectorAll('.favorite-input-image img, .favorite-model-image img').forEach(img => {
        img.addEventListener('click', (e) => {
            e.stopPropagation();
            openLightbox(img.src, img.alt || img.dataset.label || '');
        });
    });
}

function renderFavoritePromptCard(fav, data) {
    const inputImagesHtml = renderFavoriteInputImages(data.subset, data.sample_index, data.input_image_count || 1);
    const modelsHtml = renderFavoriteModelsGrid(data.subset, data.sample_index, data.models || []);

    return `
        <div class="favorite-prompt-card">
            <div class="favorite-prompt-header">
                <div class="favorite-prompt-info">
                    <div class="favorite-prompt-instruction">${escapeHtml(data.instruction || 'No instruction')}</div>
                    <div class="favorite-prompt-meta">
                        <span>Subset: ${escapeHtml(data.subset)}</span>
                        <span>Experiment: ${escapeHtml(data.exp_name)}</span>
                        <span>Index: ${data.sample_index}</span>
                        ${data.task_type ? `<span>Task: ${escapeHtml(data.task_type)}</span>` : ''}
                        ${data.prompt_source ? `<span>Source: ${escapeHtml(data.prompt_source)}</span>` : ''}
                    </div>
                </div>
                <div class="favorite-prompt-actions">
                    <button class="btn-unfavorite" data-subset="${escapeHtml(fav.subset)}" data-exp-name="${escapeHtml(fav.exp_name)}" data-sample-index="${fav.sample_index}">Remove</button>
                </div>
            </div>
            ${inputImagesHtml}
            ${modelsHtml}
        </div>
    `;
}

function renderFavoritePromptCardError(fav) {
    return `
        <div class="favorite-prompt-card">
            <div class="favorite-prompt-header">
                <div class="favorite-prompt-info">
                    <div class="favorite-prompt-instruction">${escapeHtml(fav.instruction || 'Failed to load')}</div>
                    <div class="favorite-prompt-meta">
                        <span>Subset: ${escapeHtml(fav.subset)}</span>
                        <span>Experiment: ${escapeHtml(fav.exp_name)}</span>
                        <span>Index: ${fav.sample_index}</span>
                        <span style="color: var(--accent-red);">Error loading data</span>
                    </div>
                </div>
                <div class="favorite-prompt-actions">
                    <button class="btn-unfavorite" data-subset="${escapeHtml(fav.subset)}" data-exp-name="${escapeHtml(fav.exp_name)}" data-sample-index="${fav.sample_index}">Remove</button>
                </div>
            </div>
        </div>
    `;
}

function renderFavoriteInputImages(subset, sampleIndex, count) {
    if (count === 0) return '';

    let imagesHtml = '';
    for (let i = 0; i < count; i++) {
        const imgUrl = `images/${subset}/input/${sampleIndex}/${i}`;
        imagesHtml += `
            <div class="favorite-input-image">
                <img src="${imgUrl}" alt="Input ${i + 1}" loading="lazy">
            </div>
        `;
    }

    return `
        <div class="favorite-input-section">
            <div class="favorite-input-title">Input Images (${count})</div>
            <div class="favorite-input-images">
                ${imagesHtml}
            </div>
        </div>
    `;
}

function renderFavoriteModelsGrid(subset, sampleIndex, models) {
    if (models.length === 0) {
        return `
            <div class="favorite-models-section">
                <div class="favorite-models-title">Model Outputs</div>
                <p class="placeholder">No model outputs available</p>
            </div>
        `;
    }

    const modelsHtml = models.map((m, idx) => {
        const rank = idx + 1;
        const imgUrl = `images/${subset}/${encodeURIComponent(m.model)}/${sampleIndex}`;
        const winRatePercent = (m.win_rate * 100).toFixed(1);

        let rankClass = '';
        let rankBadge = '';
        if (rank === 1) {
            rankClass = 'rank-1';
            rankBadge = '<span class="favorite-model-rank rank-1">🥇</span>';
        } else if (rank === 2) {
            rankClass = 'rank-2';
            rankBadge = '<span class="favorite-model-rank rank-2">🥈</span>';
        } else if (rank === 3) {
            rankClass = 'rank-3';
            rankBadge = '<span class="favorite-model-rank rank-3">🥉</span>';
        }

        return `
            <div class="favorite-model-card ${rankClass}">
                <div class="favorite-model-image">
                    <img src="${imgUrl}" alt="${escapeHtml(m.model)}" data-label="${escapeHtml(getModelDisplayName(m.model))}" loading="lazy">
                </div>
                <div class="favorite-model-info">
                    <div class="favorite-model-name">${rankBadge}${escapeHtml(getModelDisplayName(m.model))}</div>
                    <div class="favorite-model-stats">
                        <span class="win-rate">${winRatePercent}%</span>
                        (<span class="wins">${m.wins}W</span>/<span class="losses">${m.losses}L</span>/<span class="ties">${m.ties}T</span>)
                    </div>
                </div>
            </div>
        `;
    }).join('');

    return `
        <div class="favorite-models-section">
            <div class="favorite-models-title">Model Outputs (sorted by win rate)</div>
            <div class="favorite-models-grid">
                ${modelsHtml}
            </div>
        </div>
    `;
}

// ========== Utility Functions ==========
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Truncate text in the middle with ellipsis
function truncateMiddle(text, maxLen = 10) {
    if (!text || text.length <= maxLen) return text;
    const half = Math.floor((maxLen - 2) / 2);
    return text.slice(0, half) + '..' + text.slice(-half);
}

// ========== Event Handlers ==========
elements.subsetSelect.addEventListener('change', (e) => {
    state.subset = e.target.value || null;
    state.experiment = null;
    state.page = 1;
    state.searchQuery = '';

    // Reset all filters when switching subsets
    state.filters = { models: [], result: null, consistent: null, minImages: null, maxImages: null, promptSource: null };
    elements.resultFilter.value = '';
    elements.consistencyFilter.value = '';
    elements.resultFilterGroup.style.display = 'none';
    
    // Clear search input if exists
    if (elements.searchInput) {
        elements.searchInput.value = '';
    }

    // Hide image count filter (will be shown again if applicable in loadSubsetInfo)
    elements.imageCountFilterGroup.style.display = 'none';

    // Reset experiment dropdown
    elements.expSelect.innerHTML = '<option value="">Select experiment...</option>';
    elements.expSelect.disabled = true;

    // Clear model checkboxes
    elements.modelCheckboxes.innerHTML = '';
    updateModelCount();

    // Clear prompt source filter
    elements.promptSourceFilter.innerHTML = '<option value="">All sources</option>';
    elements.promptSourceFilterGroup.style.display = 'none';

    // Hide h2h section
    elements.h2hSection.style.display = 'none';

    // Clear both lists
    elements.battleList.innerHTML = '<div class="empty-state"><p>Select an experiment to view battles</p></div>';
    elements.promptsList.innerHTML = '<div class="empty-state"><p>Select an experiment to view prompts</p></div>';

    if (state.subset) {
        loadSubsetInfo(state.subset);
        loadEloLeaderboard();  // Load ELO rankings when subset is selected
    }
    
    syncStateToURL();
});

elements.expSelect.addEventListener('change', (e) => {
    state.experiment = e.target.value || null;
    state.page = 1;

    if (state.experiment) {
        loadCurrentView();
    }
    
    syncStateToURL();
});

// Select all / Clear all model buttons
elements.selectAllModels.addEventListener('click', () => {
    elements.modelCheckboxes.querySelectorAll('input[type="checkbox"]').forEach(cb => {
        cb.checked = true;
    });
    updateModelSelection();
});

elements.clearAllModels.addEventListener('click', () => {
    elements.modelCheckboxes.querySelectorAll('input[type="checkbox"]').forEach(cb => {
        cb.checked = false;
    });
    updateModelSelection();
});

// Image range slider handlers
elements.minImagesSlider.addEventListener('input', () => {
    const minVal = parseInt(elements.minImagesSlider.value);
    const maxVal = parseInt(elements.maxImagesSlider.value);
    if (minVal > maxVal) {
        elements.maxImagesSlider.value = minVal;
    }
    updateImageRangeDisplay();
});

elements.maxImagesSlider.addEventListener('input', () => {
    const minVal = parseInt(elements.minImagesSlider.value);
    const maxVal = parseInt(elements.maxImagesSlider.value);
    if (maxVal < minVal) {
        elements.minImagesSlider.value = maxVal;
    }
    updateImageRangeDisplay();
});

elements.applyFilters.addEventListener('click', () => {
    state.filters.models = getSelectedModels();
    state.filters.result = elements.resultFilter.value || null;
    state.filters.consistent = elements.consistencyFilter.value || null;
    state.filters.promptSource = elements.promptSourceFilter.value || null;

    // Get image range if filter is visible
    if (elements.imageCountFilterGroup.style.display !== 'none') {
        const minVal = parseInt(elements.minImagesSlider.value);
        const maxVal = parseInt(elements.maxImagesSlider.value);
        // Only set filter if it's different from the full range
        if (minVal > state.imageRange.min || maxVal < state.imageRange.max) {
            state.filters.minImages = minVal;
            state.filters.maxImages = maxVal;
        } else {
            state.filters.minImages = null;
            state.filters.maxImages = null;
        }
    } else {
        state.filters.minImages = null;
        state.filters.maxImages = null;
    }

    state.page = 1;
    loadCurrentView();
    syncStateToURL();
});

elements.clearFilters.addEventListener('click', () => {
    state.filters = { models: [], result: null, consistent: null, minImages: null, maxImages: null, promptSource: null };
    state.searchQuery = '';
    elements.modelCheckboxes.querySelectorAll('input[type="checkbox"]').forEach(cb => {
        cb.checked = false;
    });
    elements.resultFilter.value = '';
    elements.consistencyFilter.value = '';
    elements.promptSourceFilter.value = '';
    elements.resultFilterGroup.style.display = 'none';
    elements.h2hSection.style.display = 'none';
    
    // Clear search input if exists
    if (elements.searchInput) {
        elements.searchInput.value = '';
    }

    // Reset image range sliders
    if (elements.imageCountFilterGroup.style.display !== 'none') {
        elements.minImagesSlider.value = state.imageRange.min;
        elements.maxImagesSlider.value = state.imageRange.max;
        updateImageRangeDisplay();
    }

    updateModelCount();
    state.page = 1;
    loadCurrentView();
    syncStateToURL();
});

// Pagination handlers
function goToFirstPage() {
    if (state.page > 1) {
        state.page = 1;
        loadCurrentView();
        syncStateToURL();
    }
}

function goToPrevPage() {
    if (state.page > 1) {
        state.page--;
        loadCurrentView();
        syncStateToURL();
    }
}

function goToNextPage() {
    if (state.page < state.totalPages) {
        state.page++;
        loadCurrentView();
        syncStateToURL();
    }
}

function goToLastPage() {
    if (state.page < state.totalPages) {
        state.page = state.totalPages;
        loadCurrentView();
        syncStateToURL();
    }
}

elements.firstPage.addEventListener('click', goToFirstPage);
elements.prevPage.addEventListener('click', goToPrevPage);
elements.nextPage.addEventListener('click', goToNextPage);
elements.lastPage.addEventListener('click', goToLastPage);
elements.firstPageBottom.addEventListener('click', goToFirstPage);
elements.prevPageBottom.addEventListener('click', goToPrevPage);
elements.nextPageBottom.addEventListener('click', goToNextPage);
elements.lastPageBottom.addEventListener('click', goToLastPage);

// Page input handlers
function goToPage(pageNum) {
    const page = parseInt(pageNum);
    if (!isNaN(page) && page >= 1 && page <= state.totalPages && page !== state.page) {
        state.page = page;
        loadCurrentView();
        syncStateToURL();
    }
}

function handlePageInputKeydown(e) {
    if (e.key === 'Enter') {
        goToPage(e.target.value);
        e.target.value = '';
    }
}

elements.pageGo.addEventListener('click', () => {
    goToPage(elements.pageInput.value);
    elements.pageInput.value = '';
});

elements.pageGoBottom.addEventListener('click', () => {
    goToPage(elements.pageInputBottom.value);
    elements.pageInputBottom.value = '';
});

elements.pageInput.addEventListener('keydown', handlePageInputKeydown);
elements.pageInputBottom.addEventListener('keydown', handlePageInputKeydown);

// Modal handlers
elements.modalClose.addEventListener('click', hideModal);
elements.modalBackdrop.addEventListener('click', hideModal);

// Lightbox handlers
elements.lightboxClose.addEventListener('click', closeLightbox);
elements.lightbox.addEventListener('click', (e) => {
    // Close when clicking on backdrop (not the image)
    if (e.target === elements.lightbox || e.target === elements.lightboxLabel) {
        closeLightbox();
    }
});

// Favorites handlers
elements.favoritesBtn.addEventListener('click', showFavoritesModal);
elements.favoritesModalClose.addEventListener('click', hideFavoritesModal);
elements.favoritesModalBackdrop.addEventListener('click', hideFavoritesModal);
elements.clearAllFavorites.addEventListener('click', clearAllFavorites);

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Handle lightbox first (highest priority)
    if (elements.lightbox.classList.contains('active')) {
        if (e.key === 'Escape') {
            closeLightbox();
        }
        return;
    }

    // Handle model stats modal
    if (elements.modelStatsModal && !elements.modelStatsModal.classList.contains('hidden')) {
        if (e.key === 'Escape') {
            hideModelStatsModal();
        }
        return;
    }

    // Handle leaderboard modal
    if (elements.leaderboardModal && !elements.leaderboardModal.classList.contains('hidden')) {
        if (e.key === 'Escape') {
            hideLeaderboardModal();
        }
        return;
    }

    // Handle matrix modal
    if (elements.matrixModal && !elements.matrixModal.classList.contains('hidden')) {
        if (e.key === 'Escape') {
            hideMatrixModal();
        }
        return;
    }

    // Handle ELO by source modal
    if (elements.eloBySourceModal && !elements.eloBySourceModal.classList.contains('hidden')) {
        if (e.key === 'Escape') {
            hideEloBySourceModal();
        }
        return;
    }

    // Handle cross-subset modal
    if (elements.crossSubsetModal && !elements.crossSubsetModal.classList.contains('hidden')) {
        if (e.key === 'Escape') {
            hideCrossSubsetModal();
        }
        return;
    }

    // Handle ELO history modal
    if (elements.eloHistoryModal && !elements.eloHistoryModal.classList.contains('hidden')) {
        if (e.key === 'Escape') {
            hideEloHistoryModal();
        }
        return;
    }

    // Handle favorites modal
    if (!elements.favoritesModal.classList.contains('hidden')) {
        if (e.key === 'Escape') {
            hideFavoritesModal();
        }
        return;
    }

    // Handle modal
    if (!elements.modal.classList.contains('hidden')) {
        if (e.key === 'Escape') {
            hideModal();
        }
        return;
    }

    // Only when not in an input
    if (document.activeElement.tagName === 'SELECT' || document.activeElement.tagName === 'INPUT') return;

    if (e.key === 'j' || e.key === 'ArrowRight') {
        goToNextPage();
    } else if (e.key === 'k' || e.key === 'ArrowLeft') {
        goToPrevPage();
    } else if (e.key === 'Home') {
        goToFirstPage();
    } else if (e.key === 'End') {
        goToLastPage();
    } else if (e.key === 'f' || e.key === 'F') {
        showFavoritesModal();
    }
});

// ========== View Toggle Functions ==========
function switchToView(viewMode) {
    console.log('switchToView called:', viewMode);
    state.viewMode = viewMode;
    state.page = 1;

    // Update button states
    elements.viewBattlesBtn.classList.toggle('active', viewMode === 'battles');
    elements.viewPromptsBtn.classList.toggle('active', viewMode === 'prompts');

    // Toggle visibility of lists
    elements.battleList.style.display = viewMode === 'battles' ? 'flex' : 'none';
    elements.promptsList.style.display = viewMode === 'prompts' ? 'flex' : 'none';

    // Toggle visibility of battles-only filters
    document.querySelectorAll('.battles-only').forEach(el => {
        el.style.display = viewMode === 'battles' ? 'block' : 'none';
    });

    // Toggle visibility of prompts-only filters
    document.querySelectorAll('.prompts-only').forEach(el => {
        el.style.display = viewMode === 'prompts' ? 'block' : 'none';
    });

    // Hide H2H section in prompts view
    if (viewMode !== 'battles') {
        elements.h2hSection.style.display = 'none';
    }

    // Load data for the current view
    loadCurrentView();
    syncStateToURL();
}

function loadCurrentView() {
    console.log('loadCurrentView called, viewMode:', state.viewMode);
    if (state.viewMode === 'battles') {
        loadBattles();
    } else if (state.viewMode === 'prompts') {
        loadPrompts();
    }
}

// View toggle event handlers
elements.viewBattlesBtn.addEventListener('click', () => switchToView('battles'));
elements.viewPromptsBtn.addEventListener('click', () => switchToView('prompts'));

// ========== Search Functions ==========

async function performSearch() {
    const query = elements.searchInput ? elements.searchInput.value.trim() : '';
    
    if (!state.subset || !state.experiment) {
        return;
    }
    
    state.searchQuery = query;
    state.page = 1;
    
    // Show/hide clear button
    if (elements.clearSearchBtn) {
        elements.clearSearchBtn.style.display = query ? 'inline-block' : 'none';
    }
    
    loadCurrentView();
    syncStateToURL();
}

function clearSearch() {
    if (elements.searchInput) {
        elements.searchInput.value = '';
    }
    state.searchQuery = '';
    state.page = 1;
    
    if (elements.clearSearchBtn) {
        elements.clearSearchBtn.style.display = 'none';
    }
    
    loadCurrentView();
    syncStateToURL();
}

// Search event handlers
if (elements.searchBtn) {
    elements.searchBtn.addEventListener('click', performSearch);
}

if (elements.clearSearchBtn) {
    elements.clearSearchBtn.addEventListener('click', clearSearch);
}

if (elements.searchInput) {
    elements.searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            performSearch();
        }
    });
}

// ========== Prompts Model Filter Functions ==========

function renderPromptsModelCheckboxes() {
    const models = state.models || [];
    if (models.length === 0) {
        if (elements.promptsModelCheckboxes) {
            elements.promptsModelCheckboxes.innerHTML = '<p class="placeholder">No models available</p>';
        }
        return;
    }

    if (elements.promptsModelCheckboxes) {
        elements.promptsModelCheckboxes.innerHTML = models.map(model => `
            <div class="checkbox-item">
                <input type="checkbox" id="prompts-model-${escapeHtml(model)}" value="${escapeHtml(model)}"
                    ${state.promptsModelFilter.includes(model) ? 'checked' : ''}>
                <label for="prompts-model-${escapeHtml(model)}">${escapeHtml(getModelDisplayName(model))}</label>
            </div>
        `).join('');

        // Add change listeners
        elements.promptsModelCheckboxes.querySelectorAll('input[type="checkbox"]').forEach(cb => {
            cb.addEventListener('change', updatePromptsModelFilter);
        });

        updatePromptsModelCount();
    }
}

function updatePromptsModelFilter() {
    const selected = [];
    if (elements.promptsModelCheckboxes) {
        elements.promptsModelCheckboxes.querySelectorAll('input[type="checkbox"]:checked').forEach(cb => {
            selected.push(cb.value);
        });
    }
    state.promptsModelFilter = selected;
    updatePromptsModelCount();
}

function updatePromptsModelCount() {
    if (elements.promptsModelCount) {
        const count = state.promptsModelFilter.length;
        const total = state.models.length;
        elements.promptsModelCount.textContent = count > 0 ? `(${count} of ${total} selected)` : `(0 selected)`;
    }
}

function applyPromptsModelFilter() {
    state.page = 1;
    loadPrompts();
}

// Prompts model filter event handlers
if (elements.promptsSelectAllModels) {
    elements.promptsSelectAllModels.addEventListener('click', () => {
        if (elements.promptsModelCheckboxes) {
            elements.promptsModelCheckboxes.querySelectorAll('input[type="checkbox"]').forEach(cb => {
                cb.checked = true;
            });
            updatePromptsModelFilter();
            applyPromptsModelFilter();
        }
    });
}

if (elements.promptsClearAllModels) {
    elements.promptsClearAllModels.addEventListener('click', () => {
        if (elements.promptsModelCheckboxes) {
            elements.promptsModelCheckboxes.querySelectorAll('input[type="checkbox"]').forEach(cb => {
                cb.checked = false;
            });
            updatePromptsModelFilter();
            applyPromptsModelFilter();
        }
    });
}

if (elements.promptsApplyModelFilter) {
    elements.promptsApplyModelFilter.addEventListener('click', () => {
        applyPromptsModelFilter();
    });
}

// ========== Favorites Model Filter Functions ==========

function renderFavoritesModelFilter() {
    // Collect all unique models from all favorites' subsets
    const allModels = new Set();

    // Use the current subset's models as the base
    if (state.models) {
        state.models.forEach(m => allModels.add(m));
    }

    const models = Array.from(allModels).sort();

    if (models.length === 0) {
        if (elements.favoritesModelCheckboxes) {
            elements.favoritesModelCheckboxes.innerHTML = '<p class="placeholder">No models available</p>';
        }
        return;
    }

    if (elements.favoritesModelCheckboxes) {
        elements.favoritesModelCheckboxes.innerHTML = models.map(model => `
            <div class="checkbox-item${state.favoritesModelFilter.includes(model) ? ' selected' : ''}" data-model="${escapeHtml(model)}">
                <input type="checkbox" value="${escapeHtml(model)}"
                    ${state.favoritesModelFilter.includes(model) ? 'checked' : ''}>
                <span class="checkbox-label">${escapeHtml(getModelDisplayName(model))}</span>
            </div>
        `).join('');

        // Add click listeners for the entire item
        elements.favoritesModelCheckboxes.querySelectorAll('.checkbox-item').forEach(item => {
            item.addEventListener('click', () => {
                const checkbox = item.querySelector('input[type="checkbox"]');
                if (checkbox) {
                    checkbox.checked = !checkbox.checked;
                    item.classList.toggle('selected', checkbox.checked);
                    updateFavoritesModelFilter();
                }
            });
        });
    }

    // Sync the stats scope toggle checkbox with current state
    if (elements.favoritesStatsScopeAll) {
        elements.favoritesStatsScopeAll.checked = state.favoritesStatsScope === 'all';
    }
}

function updateFavoritesModelFilter() {
    const selected = [];
    if (elements.favoritesModelCheckboxes) {
        elements.favoritesModelCheckboxes.querySelectorAll('input[type="checkbox"]:checked').forEach(cb => {
            selected.push(cb.value);
        });
    }
    state.favoritesModelFilter = selected;
}

function applyFavoritesModelFilter() {
    renderFavoritesModal();
}

// Favorites model filter event handlers
if (elements.favoritesSelectAllModels) {
    elements.favoritesSelectAllModels.addEventListener('click', () => {
        if (elements.favoritesModelCheckboxes) {
            elements.favoritesModelCheckboxes.querySelectorAll('.checkbox-item').forEach(item => {
                const cb = item.querySelector('input[type="checkbox"]');
                if (cb) {
                    cb.checked = true;
                    item.classList.add('selected');
                }
            });
            updateFavoritesModelFilter();
            applyFavoritesModelFilter();
        }
    });
}

if (elements.favoritesClearAllModels) {
    elements.favoritesClearAllModels.addEventListener('click', () => {
        if (elements.favoritesModelCheckboxes) {
            elements.favoritesModelCheckboxes.querySelectorAll('.checkbox-item').forEach(item => {
                const cb = item.querySelector('input[type="checkbox"]');
                if (cb) {
                    cb.checked = false;
                    item.classList.remove('selected');
                }
            });
            updateFavoritesModelFilter();
            applyFavoritesModelFilter();
        }
    });
}

if (elements.favoritesApplyModelFilter) {
    elements.favoritesApplyModelFilter.addEventListener('click', () => {
        applyFavoritesModelFilter();
    });
}

// Stats scope toggle - controls whether win rate includes all opponents or only filtered models
if (elements.favoritesStatsScopeAll) {
    elements.favoritesStatsScopeAll.addEventListener('change', (e) => {
        state.favoritesStatsScope = e.target.checked ? 'all' : 'filtered';
        applyFavoritesModelFilter();
    });
}

// ========== ELO Leaderboard Event Handlers ==========
if (elements.viewFullLeaderboard) {
    elements.viewFullLeaderboard.addEventListener('click', () => {
        loadFullLeaderboard();
    });
}

if (elements.leaderboardModalClose) {
    elements.leaderboardModalClose.addEventListener('click', hideLeaderboardModal);
}

if (elements.leaderboardModalBackdrop) {
    elements.leaderboardModalBackdrop.addEventListener('click', hideLeaderboardModal);
}

if (elements.modelStatsModalClose) {
    elements.modelStatsModalClose.addEventListener('click', hideModelStatsModal);
}

if (elements.modelStatsModalBackdrop) {
    elements.modelStatsModalBackdrop.addEventListener('click', hideModelStatsModal);
}

// ========== URL State Management ==========

/**
 * Sync current state to URL query parameters
 * Called after any state change that should be persisted
 */
function syncStateToURL() {
    const params = new URLSearchParams();
    
    // Gallery page - include subset and experiment
    if (state.currentPage === 'gallery' && state.subset) {
        params.set('subset', state.subset);
        if (state.experiment) params.set('exp', state.experiment);
        
        // View mode (only if not default)
        if (state.viewMode !== 'battles') params.set('view', state.viewMode);
        
        // Page number (only if not 1)
        if (state.page > 1) params.set('p', state.page);
        
        // Search query
        if (state.searchQuery) params.set('q', state.searchQuery);
        
        // Filters
        if (state.filters.models && state.filters.models.length > 0) {
            params.set('models', state.filters.models.join(','));
        }
        if (state.filters.result) params.set('result', state.filters.result);
        if (state.filters.consistent !== null) params.set('consistent', state.filters.consistent);
        if (state.filters.minImages !== null) params.set('minImg', state.filters.minImages);
        if (state.filters.maxImages !== null) params.set('maxImg', state.filters.maxImages);
        if (state.filters.promptSource) params.set('source', state.filters.promptSource);
    }
    // Overview page has no params (default)
    
    // Build URL
    const newURL = params.toString() ? `?${params}` : window.location.pathname;
    window.history.replaceState({}, '', newURL);
}

/**
 * Load state from URL query parameters on page load
 * Returns true if there was state to restore
 */
function loadStateFromURL() {
    const params = new URLSearchParams(window.location.search);
    
    // Check for gallery page (has subset param)
    if (params.has('subset')) {
        window._urlStateToRestore = {
            currentPage: 'gallery',
            subset: params.get('subset'),
            experiment: params.get('exp'),
            viewMode: params.get('view') || 'battles',
            page: parseInt(params.get('p')) || 1,
            searchQuery: params.get('q') || '',
            filters: {
                models: params.get('models') ? params.get('models').split(',') : [],
                result: params.get('result') || null,
                consistent: params.has('consistent') ? params.get('consistent') : null,
                minImages: params.has('minImg') ? parseInt(params.get('minImg')) : null,
                maxImages: params.has('maxImg') ? parseInt(params.get('maxImg')) : null,
                promptSource: params.get('source') || null,
            }
        };
        return true;
    }
    
    // Default to overview (no state to restore)
    return false;
}

/**
 * Apply URL state after subset info has loaded
 */
async function applyURLState() {
    const urlState = window._urlStateToRestore;
    if (!urlState) return;
    
    // Handle gallery page
    if (urlState.currentPage === 'gallery' && urlState.subset) {
        // Switch to gallery page first
        switchToPage('gallery');
        
        elements.subsetSelect.value = urlState.subset;
        state.subset = urlState.subset;
        await loadSubsetInfo(urlState.subset);
        await loadEloLeaderboard();
        
        // Set experiment
        if (urlState.experiment) {
            elements.expSelect.value = urlState.experiment;
            state.experiment = urlState.experiment;
        }
        
        // Set view mode
        if (urlState.viewMode && urlState.viewMode !== 'battles') {
            state.viewMode = urlState.viewMode;
            elements.viewBattlesBtn.classList.toggle('active', urlState.viewMode === 'battles');
            elements.viewPromptsBtn.classList.toggle('active', urlState.viewMode === 'prompts');
            elements.battleList.style.display = urlState.viewMode === 'battles' ? 'flex' : 'none';
            elements.promptsList.style.display = urlState.viewMode === 'prompts' ? 'flex' : 'none';
        }
        
        // Set page
        state.page = urlState.page || 1;
        
        // Set search query
        if (urlState.searchQuery) {
            state.searchQuery = urlState.searchQuery;
            if (elements.searchInput) {
                elements.searchInput.value = urlState.searchQuery;
            }
        }
        
        // Set filters
        if (urlState.filters.models && urlState.filters.models.length > 0) {
            state.filters.models = urlState.filters.models;
            // Check the corresponding checkboxes
            elements.modelCheckboxes.querySelectorAll('input[type="checkbox"]').forEach(cb => {
                cb.checked = urlState.filters.models.includes(cb.value);
            });
            updateModelCount();
            updateModelSelection();
        }
        
        if (urlState.filters.result) {
            state.filters.result = urlState.filters.result;
            elements.resultFilter.value = urlState.filters.result;
        }
        
        if (urlState.filters.consistent !== null) {
            state.filters.consistent = urlState.filters.consistent;
            elements.consistencyFilter.value = urlState.filters.consistent;
        }
        
        if (urlState.filters.promptSource) {
            state.filters.promptSource = urlState.filters.promptSource;
            elements.promptSourceFilter.value = urlState.filters.promptSource;
        }
        
        if (urlState.filters.minImages !== null) {
            state.filters.minImages = urlState.filters.minImages;
            elements.minImagesSlider.value = urlState.filters.minImages;
        }
        
        if (urlState.filters.maxImages !== null) {
            state.filters.maxImages = urlState.filters.maxImages;
            elements.maxImagesSlider.value = urlState.filters.maxImages;
        }
        
        updateImageRangeDisplay();
        
        // Load data if experiment is set
        if (state.experiment) {
            loadCurrentView();
        }
    }
    
    // Clear the stored state
    window._urlStateToRestore = null;
}

// ========== Win Rate Matrix Modal Functions ==========

async function loadWinRateMatrix() {
    if (!state.subset) return;
    
    try {
        elements.matrixContent.innerHTML = '<div class="loading">Loading matrix...</div>';
        elements.matrixSubsetName.textContent = state.subset;
        showMatrixModal();
        
        const data = await fetchJSON(`api/subsets/${state.subset}/matrix`);
        renderWinRateMatrix(data);
    } catch (error) {
        console.error('Failed to load win rate matrix:', error);
        elements.matrixContent.innerHTML = '<div class="empty-state"><p>Failed to load matrix data</p></div>';
    }
}

function renderWinRateMatrix(data) {
    const { models, matrix, counts: battle_counts } = data;
    
    if (!models || models.length === 0) {
        elements.matrixContent.innerHTML = '<div class="empty-state"><p>No model data available</p></div>';
        return;
    }
    
    // Build the matrix table
    // Header row with model names
    let headerCells = '<th class="matrix-corner"></th>';
    models.forEach((model, idx) => {
        headerCells += `<th class="matrix-header-cell" title="${escapeHtml(model)}">${escapeHtml(truncateMiddle(getModelDisplayName(model), 8))}</th>`;
    });
    
    // Body rows
    let bodyRows = '';
    models.forEach((rowModel, rowIdx) => {
        let cells = `<td class="matrix-row-header">${escapeHtml(getModelDisplayName(rowModel))}</td>`;
        
        models.forEach((colModel, colIdx) => {
            if (rowIdx === colIdx) {
                // Diagonal - same model
                cells += '<td class="matrix-cell matrix-diagonal">-</td>';
            } else {
                const winRate = matrix[rowIdx][colIdx];
                const battleCount = battle_counts[rowIdx][colIdx];
                
                if (battleCount === 0) {
                    cells += '<td class="matrix-cell matrix-no-data" title="No battles">-</td>';
                } else {
                    // Color based on win rate: red (0%) -> white (50%) -> green (100%)
                    const bgColor = getWinRateColor(winRate);
                    const textColor = getWinRateTextColor(winRate);
                    const winRatePercent = (winRate * 100).toFixed(1);
                    cells += `<td class="matrix-cell" style="background-color: ${bgColor}; color: ${textColor}" title="${escapeHtml(getModelDisplayName(rowModel))} vs ${escapeHtml(getModelDisplayName(colModel))}: ${winRatePercent}% (${battleCount} battles)">${winRatePercent}%</td>`;
                }
            }
        });
        
        bodyRows += `<tr>${cells}</tr>`;
    });
    
    elements.matrixContent.innerHTML = `
        <div class="matrix-scroll-container">
            <table class="win-rate-matrix">
                <thead><tr>${headerCells}</tr></thead>
                <tbody>${bodyRows}</tbody>
            </table>
        </div>
        <div class="matrix-legend">
            <span class="matrix-legend-label">Row model win rate vs column model:</span>
            <div class="matrix-legend-gradient">
                <span class="legend-low">0%</span>
                <div class="legend-bar"></div>
                <span class="legend-high">100%</span>
            </div>
        </div>
    `;
}

function getWinRateColor(winRate) {
    // Red (0%) -> White (50%) -> Green (100%)
    if (winRate < 0.5) {
        // Red to white
        const intensity = winRate * 2; // 0 to 1
        const r = 255;
        const g = Math.round(200 * intensity + 55);
        const b = Math.round(200 * intensity + 55);
        return `rgb(${r}, ${g}, ${b})`;
    } else {
        // White to green
        const intensity = (winRate - 0.5) * 2; // 0 to 1
        const r = Math.round(255 * (1 - intensity * 0.6));
        const g = Math.round(255 - intensity * 55);
        const b = Math.round(255 * (1 - intensity * 0.6));
        return `rgb(${r}, ${g}, ${b})`;
    }
}

function getWinRateTextColor(winRate) {
    // Use dark text for light backgrounds (near 50%), white text for strong red (near 0%)
    // Near 0%: strong red background -> white text
    // Near 50%: white/light background -> black text
    // Near 100%: green background -> black text (green is not too dark)
    if (winRate < 0.25) {
        // Strong red - use white text
        return '#fff';
    } else {
        // Light red, white, or green - use black text
        return '#000';
    }
}

function showMatrixModal() {
    if (elements.matrixModal) {
        elements.matrixModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }
}

function hideMatrixModal() {
    if (elements.matrixModal) {
        elements.matrixModal.classList.add('hidden');
        document.body.style.overflow = '';
    }
}

// ========== ELO by Source Modal Functions ==========

async function loadEloBySource() {
    if (!state.subset) return;
    
    try {
        elements.eloBySourceContent.innerHTML = '<div class="loading">Loading ELO by source...</div>';
        elements.eloBySourceSubsetName.textContent = state.subset;
        showEloBySourceModal();
        
        const data = await fetchJSON(`api/subsets/${state.subset}/leaderboard/by-source`);
        renderEloBySource(data);
    } catch (error) {
        console.error('Failed to load ELO by source:', error);
        elements.eloBySourceContent.innerHTML = '<div class="empty-state"><p>Failed to load ELO by source data</p></div>';
    }
}

function renderEloBySource(data) {
    const { sources, leaderboards, battle_counts } = data;
    
    if (!sources || sources.length === 0) {
        elements.eloBySourceContent.innerHTML = '<div class="empty-state"><p>No source-specific ELO data available</p></div>';
        return;
    }
    
    // sources is an array of source names (sorted by battle count)
    const sourceNames = sources;
    
    const sectionsHtml = sourceNames.map(sourceName => {
        const leaderboard = leaderboards[sourceName] || [];
        const battleCount = battle_counts[sourceName] || 0;
        
        if (leaderboard.length === 0) {
            return `
                <div class="source-section">
                    <div class="source-section-header" onclick="this.parentElement.classList.toggle('expanded')">
                        <span class="source-name">${escapeHtml(sourceName)}</span>
                        <span class="source-battles">(${battleCount} battles)</span>
                        <span class="expand-icon">▼</span>
                    </div>
                    <div class="source-section-content">
                        <p class="placeholder">No ELO data for this source</p>
                    </div>
                </div>
            `;
        }
        
        const tableRows = leaderboard.map((model, index) => {
            const rank = index + 1;
            const rankClass = rank <= 3 ? `rank-${rank}` : '';
            const winRatePercent = (model.win_rate * 100).toFixed(1);
            return `
                <tr>
                    <td class="rank-cell ${rankClass}">#${rank}</td>
                    <td class="model-cell">${escapeHtml(getModelDisplayName(model.model))}</td>
                    <td class="elo-cell">${Math.round(model.elo)}</td>
                    <td class="stat-cell wins">${model.wins}</td>
                    <td class="stat-cell losses">${model.losses}</td>
                    <td class="stat-cell ties">${model.ties}</td>
                    <td class="win-rate-cell">${winRatePercent}%</td>
                </tr>
            `;
        }).join('');
        
        return `
            <div class="source-section expanded">
                <div class="source-section-header" onclick="this.parentElement.classList.toggle('expanded')">
                    <span class="source-name">${escapeHtml(sourceName)}</span>
                    <span class="source-battles">(${battleCount} battles, ${leaderboard.length} models)</span>
                    <span class="expand-icon">▼</span>
                </div>
                <div class="source-section-content">
                    <table class="source-leaderboard">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Model</th>
                                <th>ELO</th>
                                <th>W</th>
                                <th>L</th>
                                <th>T</th>
                                <th>Win %</th>
                            </tr>
                        </thead>
                        <tbody>${tableRows}</tbody>
                    </table>
                </div>
            </div>
        `;
    }).join('');
    
    elements.eloBySourceContent.innerHTML = `
        <div class="source-sections-container">
            ${sectionsHtml}
        </div>
    `;
}

function showEloBySourceModal() {
    if (elements.eloBySourceModal) {
        elements.eloBySourceModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }
}

function hideEloBySourceModal() {
    if (elements.eloBySourceModal) {
        elements.eloBySourceModal.classList.add('hidden');
        document.body.style.overflow = '';
    }
}

// ========== ELO History Modal Functions ==========

let eloHistoryState = {
    data: null,
    visibleModels: new Set(),
    granularity: 10,
};

async function loadEloHistory() {
    if (!state.subset) return;
    
    try {
        const granularity = elements.eloHistoryGranularity ? elements.eloHistoryGranularity.value || 'experiment' : 'experiment';
        eloHistoryState.granularity = granularity;
        
        elements.eloHistoryContent.innerHTML = '<div class="loading">Loading ELO history...</div>';
        elements.eloHistoryLegend.innerHTML = '';
        showEloHistoryModal();
        
        const rawData = await fetchJSON(`api/subsets/${state.subset}/elo-history?granularity=${granularity}`);
        
        // Transform backend format to frontend format
        // Backend: { timestamps: [], models: { model -> [elo values] }, battle_counts: [] }
        // Frontend: { history: [{ timestamp, elos: {model -> elo} }], models: [list of model names] }
        const { timestamps, models: modelElos, battle_counts } = rawData;
        const modelNames = Object.keys(modelElos);
        const history = timestamps.map((timestamp, i) => {
            const elos = {};
            for (const model of modelNames) {
                const eloValue = modelElos[model][i];
                if (eloValue !== null && eloValue !== undefined) {
                    elos[model] = eloValue;
                }
            }
            return { timestamp, elos, battle_count: battle_counts[i] };
        });
        
        const data = { history, models: modelNames };
        eloHistoryState.data = data;
        
        // Initialize all models as visible
        eloHistoryState.visibleModels = new Set(data.models || []);
        
        renderEloHistory();
        renderEloHistoryLegend();
    } catch (error) {
        console.error('Failed to load ELO history:', error);
        elements.eloHistoryContent.innerHTML = '<div class="empty-state"><p>Failed to load ELO history</p></div>';
    }
}

function renderEloHistory() {
    const data = eloHistoryState.data;
    if (!data || !data.history || data.history.length === 0) {
        elements.eloHistoryContent.innerHTML = '<div class="empty-state"><p>No ELO history available</p></div>';
        return;
    }
    
    const { history, models } = data;
    
    // Filter to visible models
    const visibleModels = models.filter(m => eloHistoryState.visibleModels.has(m));
    if (visibleModels.length === 0) {
        elements.eloHistoryContent.innerHTML = '<div class="empty-state"><p>No models selected. Click on models in the legend to show them.</p></div>';
        return;
    }
    
    // Calculate bounds
    let minElo = Infinity, maxElo = -Infinity;
    history.forEach(point => {
        visibleModels.forEach(model => {
            const elo = point.elos[model];
            if (elo !== undefined) {
                minElo = Math.min(minElo, elo);
                maxElo = Math.max(maxElo, elo);
            }
        });
    });
    
    // Add padding
    const eloPadding = (maxElo - minElo) * 0.1 || 50;
    minElo -= eloPadding;
    maxElo += eloPadding;
    
    // SVG dimensions
    const width = 800;
    const height = 400;
    const margin = { top: 20, right: 120, bottom: 50, left: 60 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    
    // Scales
    const xScale = (i) => margin.left + (i / (history.length - 1)) * plotWidth;
    const yScale = (elo) => margin.top + (1 - (elo - minElo) / (maxElo - minElo)) * plotHeight;
    
    // Generate colors for models
    const colors = generateModelColors(models.length);
    const modelColorMap = {};
    models.forEach((model, i) => {
        modelColorMap[model] = colors[i];
    });
    
    // Build SVG paths and points for each visible model
    let pathsHtml = '';
    let pointsHtml = '';
    visibleModels.forEach(model => {
        const color = modelColorMap[model];
        let pathData = '';
        let started = false;
        
        history.forEach((point, i) => {
            const elo = point.elos[model];
            if (elo !== undefined) {
                const x = xScale(i);
                const y = yScale(elo);
                if (!started) {
                    pathData += `M ${x} ${y}`;
                    started = true;
                } else {
                    pathData += ` L ${x} ${y}`;
                }
                // Add interactive point
                const eloRounded = Math.round(elo);
                const timestamp = point.timestamp || '';
                pointsHtml += `<circle cx="${x}" cy="${y}" r="4" fill="${color}" class="elo-point" data-model="${escapeHtml(model)}" data-elo="${eloRounded}" data-timestamp="${escapeHtml(timestamp)}" data-display-name="${escapeHtml(getModelDisplayName(model))}"/>`;
            }
        });
        
        if (pathData) {
            pathsHtml += `<path d="${pathData}" stroke="${color}" fill="none" stroke-width="2" class="elo-line" data-model="${escapeHtml(model)}"/>`;
        }
    });
    
    // X-axis labels (show all experiment names)
    let xAxisHtml = '';
    history.forEach((point, i) => {
        const x = xScale(i);
        const label = point.timestamp || '';
        xAxisHtml += `<text x="${x}" y="${height - margin.bottom + 15}" text-anchor="end" class="axis-label" transform="rotate(-45, ${x}, ${height - margin.bottom + 15})">${escapeHtml(label)}</text>`;
    });
    
    // Y-axis labels (ELO values)
    let yAxisHtml = '';
    const eloStep = Math.ceil((maxElo - minElo) / 5);
    for (let elo = Math.ceil(minElo); elo <= maxElo; elo += eloStep) {
        const y = yScale(elo);
        yAxisHtml += `<text x="${margin.left - 10}" y="${y + 4}" text-anchor="end" class="axis-label">${Math.round(elo)}</text>`;
        yAxisHtml += `<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" class="grid-line"/>`;
    }
    
    // Determine X-axis title based on granularity
    const xAxisTitle = eloHistoryState.granularity === 'experiment' ? 'Experiment' : 'Time';
    
    elements.eloHistoryContent.innerHTML = `
        <div class="elo-history-chart-container">
            <svg width="100%" height="${height}" viewBox="0 0 ${width} ${height}" class="elo-history-chart">
                <!-- Grid lines -->
                ${yAxisHtml}
                
                <!-- Axes -->
                <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" class="axis-line"/>
                <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" class="axis-line"/>
                
                <!-- X-axis labels -->
                ${xAxisHtml}
                <text x="${width / 2}" y="${height - 2}" text-anchor="middle" class="axis-title">${xAxisTitle}</text>
                
                <!-- Y-axis title -->
                <text x="15" y="${height / 2}" text-anchor="middle" transform="rotate(-90, 15, ${height / 2})" class="axis-title">ELO</text>
                
                <!-- Data lines -->
                ${pathsHtml}
                
                <!-- Interactive points -->
                ${pointsHtml}
            </svg>
            <div class="elo-tooltip" id="elo-tooltip"></div>
        </div>
    `;
    
    // Add tooltip event listeners
    const tooltip = document.getElementById('elo-tooltip');
    const points = elements.eloHistoryContent.querySelectorAll('.elo-point');
    points.forEach(point => {
        point.addEventListener('mouseenter', (e) => {
            const model = point.getAttribute('data-model');
            const elo = point.getAttribute('data-elo');
            const timestamp = point.getAttribute('data-timestamp');
            tooltip.innerHTML = `<strong>${model}</strong><br>ELO: ${elo}<br>${timestamp}`;
            tooltip.classList.add('visible');
        });
        point.addEventListener('mousemove', (e) => {
            const container = elements.eloHistoryContent.querySelector('.elo-history-chart-container');
            const rect = container.getBoundingClientRect();
            tooltip.style.left = (e.clientX - rect.left + 10) + 'px';
            tooltip.style.top = (e.clientY - rect.top - 10) + 'px';
        });
        point.addEventListener('mouseleave', () => {
            tooltip.classList.remove('visible');
        });
    });
}

function renderEloHistoryLegend() {
    const data = eloHistoryState.data;
    if (!data || !data.models) return;
    
    const { models } = data;
    const colors = generateModelColors(models.length);
    
    elements.eloHistoryLegend.innerHTML = models.map((model, i) => {
        const isVisible = eloHistoryState.visibleModels.has(model);
        return `
            <div class="legend-item ${isVisible ? '' : 'hidden-model'}" data-model="${escapeHtml(model)}">
                <span class="legend-color" style="background-color: ${colors[i]}"></span>
                <span class="legend-label">${escapeHtml(truncateMiddle(getModelDisplayName(model), 15))}</span>
            </div>
        `;
    }).join('');
    
    // Add click handlers to toggle visibility
    elements.eloHistoryLegend.querySelectorAll('.legend-item').forEach(item => {
        item.addEventListener('click', () => {
            const model = item.dataset.model;
            if (eloHistoryState.visibleModels.has(model)) {
                eloHistoryState.visibleModels.delete(model);
                item.classList.add('hidden-model');
            } else {
                eloHistoryState.visibleModels.add(model);
                item.classList.remove('hidden-model');
            }
            renderEloHistory();
        });
    });
}

function generateModelColors(count) {
    const colors = [];
    for (let i = 0; i < count; i++) {
        const hue = (i * 360 / count) % 360;
        colors.push(`hsl(${hue}, 70%, 50%)`);
    }
    return colors;
}

function showEloHistoryModal() {
    if (elements.eloHistoryModal) {
        elements.eloHistoryModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }
}

function hideEloHistoryModal() {
    if (elements.eloHistoryModal) {
        elements.eloHistoryModal.classList.add('hidden');
        document.body.style.overflow = '';
    }
}

// ========== Event Handlers for New Modals ==========

// Matrix modal
if (elements.viewMatrixBtn) {
    elements.viewMatrixBtn.addEventListener('click', loadWinRateMatrix);
}
if (elements.matrixModalClose) {
    elements.matrixModalClose.addEventListener('click', hideMatrixModal);
}
if (elements.matrixModalBackdrop) {
    elements.matrixModalBackdrop.addEventListener('click', hideMatrixModal);
}

// ELO by Source modal
if (elements.viewEloBySourceBtn) {
    elements.viewEloBySourceBtn.addEventListener('click', loadEloBySource);
}
if (elements.eloBySourceModalClose) {
    elements.eloBySourceModalClose.addEventListener('click', hideEloBySourceModal);
}
if (elements.eloBySourceModalBackdrop) {
    elements.eloBySourceModalBackdrop.addEventListener('click', hideEloBySourceModal);
}

// Cross-Subset page event handlers
if (elements.crossSubsetSelectAll) {
    elements.crossSubsetSelectAll.addEventListener('click', () => {
        state.crossSubsetState.subsets.forEach(s => state.crossSubsetState.selectedSubsets.add(s));
        renderCrossSubsetCheckboxes();
        updateCrossSubsetInfo();
    });
}
if (elements.crossSubsetClearAll) {
    elements.crossSubsetClearAll.addEventListener('click', () => {
        state.crossSubsetState.selectedSubsets.clear();
        renderCrossSubsetCheckboxes();
        updateCrossSubsetInfo();
    });
}
if (elements.calculateMergedElo) {
    elements.calculateMergedElo.addEventListener('click', calculateMergedEloForPage);
}

// Navigation event handlers
if (elements.logoLink) {
    elements.logoLink.addEventListener('click', () => switchToPage('overview'));
}
if (elements.navOverview) {
    elements.navOverview.addEventListener('click', () => switchToPage('overview'));
}
if (elements.navGallery) {
    elements.navGallery.addEventListener('click', () => switchToPage('gallery'));
}

// Cross-Subset modal
if (elements.crossSubsetBtn) {
    elements.crossSubsetBtn.addEventListener('click', showCrossSubsetModal);
}
if (elements.crossSubsetModalClose) {
    elements.crossSubsetModalClose.addEventListener('click', hideCrossSubsetModal);
}
if (elements.crossSubsetModalBackdrop) {
    elements.crossSubsetModalBackdrop.addEventListener('click', hideCrossSubsetModal);
}

// ELO History modal
if (elements.viewEloHistoryBtn) {
    elements.viewEloHistoryBtn.addEventListener('click', loadEloHistory);
}
if (elements.eloHistoryModalClose) {
    elements.eloHistoryModalClose.addEventListener('click', hideEloHistoryModal);
}
if (elements.eloHistoryModalBackdrop) {
    elements.eloHistoryModalBackdrop.addEventListener('click', hideEloHistoryModal);
}
if (elements.eloHistoryGranularity) {
    elements.eloHistoryGranularity.addEventListener('change', loadEloHistory);
}

// ========== Initialize ==========
loadFavoritesFromStorage();

// Load model aliases first
loadModelAliases();

// Check for URL state first
const hasURLState = loadStateFromURL();

// Load subsets, then apply URL state or show overview
loadSubsets().then(() => {
    if (hasURLState) {
        applyURLState();
    } else {
        // Default to overview page
        switchToPage('overview');
    }
});
