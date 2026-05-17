document.addEventListener('DOMContentLoaded', () => {
    // State
    let currentData = [];
    let currentFilters = { categories: [], stock_status: [] };
    let chatHistory = [];

    // DOM Elements
    const loadingIndicator = document.getElementById('loadingIndicator');
    const categoryFilter = document.getElementById('categoryFilter');
    const stockFilter = document.getElementById('stockFilter');
    const applyFiltersBtn = document.getElementById('applyFiltersBtn');
    
    // KPIs
    const kpiTotal = document.getElementById('kpiTotal');
    const kpiAvgPrice = document.getElementById('kpiAvgPrice');
    const kpiAvgScore = document.getElementById('kpiAvgScore');
    
    // Tables
    const rulesTableBody = document.querySelector('#rulesTable tbody');
    const productsTableHead = document.querySelector('#productsTable document'); // Will fix below
    const productsTable = document.getElementById('productsTable');
    
    // Chat
    const chatWidget = document.getElementById('chatWidget');
    const chatHeader = document.getElementById('chatHeader');
    const toggleChatBtn = document.getElementById('toggleChatBtn');
    const chatMessages = document.getElementById('chatMessages');
    const chatInput = document.getElementById('chatInput');
    const sendChatBtn = document.getElementById('sendChatBtn');

    // Utility: Format numbers
    const formatNumber = (num, decimals = 2) => {
        return Number(num).toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
    };

    // Fetch data
    const loadDashboardData = async (filters = {}) => {
        showLoading(true);
        try {
            const response = await fetch('/api/data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(filters)
            });
            
            if (!response.ok) throw new Error('Failed to load data');
            
            const result = await response.json();
            
            // Populate filters only on initial load
            if (Object.keys(filters).length === 0) {
                populateSelect(categoryFilter, result.filter_options.categories);
                populateSelect(stockFilter, result.filter_options.stock_status);
            }
            
            currentData = result.data;
            updateKPIs(result.kpis);
            renderCharts(result.data);
            renderRulesTable(result.association_rules);
            renderDataTable(result.data);
            
        } catch (error) {
            console.error(error);
            alert("Error loading dashboard data.");
        } finally {
            showLoading(false);
        }
    };

    const showLoading = (show) => {
        if (show) loadingIndicator.classList.remove('hidden');
        else loadingIndicator.classList.add('hidden');
    };

    const populateSelect = (selectElement, options) => {
        selectElement.innerHTML = '';
        options.forEach(opt => {
            const el = document.createElement('option');
            el.value = opt;
            el.textContent = opt;
            selectElement.appendChild(el);
        });
    };

    const getSelectedValues = (selectElement) => {
        return Array.from(selectElement.selectedOptions).map(opt => opt.value);
    };

    // Update KPIs
    const updateKPIs = (kpis) => {
        kpiTotal.textContent = formatNumber(kpis.total_products, 0);
        kpiAvgPrice.textContent = `$${formatNumber(kpis.avg_price)}`;
        kpiAvgScore.textContent = formatNumber(kpis.avg_score, 3);
    };

    // Render Charts (Plotly)
    const renderCharts = (data) => {
        // Prepare Plotly layout template
        const layoutTemplate = {
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#94a3b8', family: "'Plus Jakarta Sans', sans-serif" },
            margin: { t: 20, r: 20, l: 50, b: 40 },
            xaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
            yaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
        };

        // 1. Price vs Score
        const validPriceScore = data.filter(d => d.price !== 'N/A' && d.final_score !== 'N/A');
        
        const priceScoreTrace = {
            x: validPriceScore.map(d => d.price),
            y: validPriceScore.map(d => d.final_score),
            mode: 'markers',
            type: 'scatter',
            text: validPriceScore.map(d => d.name),
            marker: {
                size: 10,
                color: validPriceScore.map(d => d.cluster_id),
                colorscale: 'Viridis',
                opacity: 0.8,
                line: { width: 1, color: 'rgba(255,255,255,0.2)' }
            }
        };

        Plotly.newPlot('scatterPlot', [priceScoreTrace], {
            ...layoutTemplate,
            xaxis: { ...layoutTemplate.xaxis, title: 'Price ($)' },
            yaxis: { ...layoutTemplate.yaxis, title: 'Final Score' }
        }, {responsive: true});

        // 2. PCA Clusters
        const validPca = data.filter(d => d.pca_1 !== 'N/A' && d.pca_2 !== 'N/A');
        
        const pcaTrace = {
            x: validPca.map(d => d.pca_1),
            y: validPca.map(d => d.pca_2),
            mode: 'markers',
            type: 'scatter',
            text: validPca.map(d => d.name),
            marker: {
                size: 10,
                color: validPca.map(d => d.cluster_id),
                colorscale: 'Plasma',
                opacity: 0.8,
                line: { width: 1, color: 'rgba(255,255,255,0.2)' }
            }
        };

        Plotly.newPlot('pcaPlot', [pcaTrace], {
            ...layoutTemplate,
            xaxis: { ...layoutTemplate.xaxis, title: 'PCA 1' },
            yaxis: { ...layoutTemplate.yaxis, title: 'PCA 2' }
        }, {responsive: true});
    };

    // Render Rules Table
    const renderRulesTable = (rules) => {
        rulesTableBody.innerHTML = '';
        if (!rules || rules.length === 0) {
            rulesTableBody.innerHTML = '<tr><td colspan="3" style="text-align:center">No rules available</td></tr>';
            return;
        }

        rules.forEach(rule => {
            const ant = (rule.antecedents_names || rule.antecedents || []).join(', ');
            const con = (rule.consequents_names || rule.consequents || []).join(', ');
            const conf = (rule.confidence || 0) * 100;
            const lift = (rule.lift || 0);

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>Si un client achète <strong>[${ant}]</strong>, il achète <strong>[${con}]</strong></td>
                <td>${conf.toFixed(1)}%</td>
                <td>${lift.toFixed(2)}</td>
            `;
            rulesTableBody.appendChild(tr);
        });
    };

    // Render Data Table
    const renderDataTable = (data) => {
        const thead = productsTable.querySelector('thead');
        const tbody = productsTable.querySelector('tbody');
        
        thead.innerHTML = '';
        tbody.innerHTML = '';

        if (data.length === 0) return;

        // Headers
        const columns = Object.keys(data[0]).filter(col => !col.startsWith('Unnamed'));
        const trHead = document.createElement('tr');
        columns.forEach(col => {
            const th = document.createElement('th');
            th.textContent = col.replace(/_/g, ' ');
            trHead.appendChild(th);
        });
        thead.appendChild(trHead);

        // Rows (limit to 50 for performance)
        data.slice(0, 50).forEach(row => {
            const tr = document.createElement('tr');
            columns.forEach(col => {
                const td = document.createElement('td');
                const val = row[col];
                // Format if number
                if (typeof val === 'number' && !Number.isInteger(val)) {
                    td.textContent = val.toFixed(2);
                } else {
                    td.textContent = val;
                }
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
    };

    // Event Listeners
    applyFiltersBtn.addEventListener('click', () => {
        currentFilters = {
            categories: getSelectedValues(categoryFilter),
            stock_status: getSelectedValues(stockFilter)
        };
        loadDashboardData(currentFilters);
    });

    // Chat logic
    const toggleChat = () => {
        chatWidget.classList.toggle('collapsed');
    };

    chatHeader.addEventListener('click', toggleChat);
    toggleChatBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleChat();
    });

    const addChatMessage = (text, sender) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}`;
        msgDiv.textContent = text;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        if (sender !== 'system') {
            chatHistory.push({ role: sender, content: text });
        }
    };

    const handleChatSubmit = async () => {
        const text = chatInput.value.trim();
        if (!text) return;

        addChatMessage(text, 'user');
        chatInput.value = '';
        
        // Disable input
        chatInput.disabled = true;
        sendChatBtn.disabled = true;
        
        // Add loading indicator message
        const loadingId = 'loading-' + Date.now();
        const loadingDiv = document.createElement('div');
        loadingDiv.id = loadingId;
        loadingDiv.className = `message assistant`;
        loadingDiv.innerHTML = '<i class="fa-solid fa-ellipsis fa-fade"></i>';
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: text,
                    filters: currentFilters,
                    history: chatHistory.slice(0, -1) // Excluding the message just added
                })
            });
            
            const data = await response.json();
            
            // Remove loading indicator
            document.getElementById(loadingId).remove();
            
            if (data.reply) {
                addChatMessage(data.reply, 'assistant');
            } else {
                addChatMessage("Error: Could not get a response.", 'assistant');
            }
        } catch (error) {
            document.getElementById(loadingId).remove();
            addChatMessage("Network error. Please try again.", 'assistant');
        } finally {
            chatInput.disabled = false;
            sendChatBtn.disabled = false;
            chatInput.focus();
        }
    };

    sendChatBtn.addEventListener('click', handleChatSubmit);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleChatSubmit();
    });

    // Initial load
    loadDashboardData();
});