let currentReportType = 'overview';
let reportData = null;
let charts = {};

const API_URLS = {
    reportData: '/dashboard/api/reports/data/',
    exportPDF: '/dashboard/api/reports/export-pdf/',
    exportCSV: '/dashboard/api/reports/export-csv/'
};

document.addEventListener('DOMContentLoaded', function() {
    console.log('[DEBUG] Reports page loaded');
    initializeReports();
    setupEventListeners();
    loadReportData();
});

function initializeReports() {
    const reportTypes = document.querySelectorAll('.report-type-card');
    reportTypes.forEach(card => {
        card.addEventListener('click', function() {
            reportTypes.forEach(c => c.classList.remove('active'));
            this.classList.add('active');
            currentReportType = this.dataset.type;
            console.log('[DEBUG] Report type changed to:', currentReportType);
            renderReport();
        });
    });
}

function setupEventListeners() {
    document.getElementById('applyFiltersBtn')?.addEventListener('click', applyFilters);
    document.getElementById('resetFiltersBtn')?.addEventListener('click', resetFilters);
    document.getElementById('exportPdfBtn')?.addEventListener('click', exportToPDF);
    document.getElementById('exportCsvBtn')?.addEventListener('click', exportToCSV);
    document.getElementById('printReportBtn')?.addEventListener('click', printReport);
}

function applyFilters() {
    const filters = {
        dateFrom: document.getElementById('dateFrom')?.value,
        dateTo: document.getElementById('dateTo')?.value,
        riskLevel: document.getElementById('riskLevel')?.value,
        department: document.getElementById('department')?.value
    };
    
    console.log('[DEBUG] Applying filters:', filters);
    loadReportData(filters);
}

function resetFilters() {
    document.getElementById('dateFrom').value = '';
    document.getElementById('dateTo').value = '';
    document.getElementById('riskLevel').value = 'all';
    document.getElementById('department').value = 'all';
    
    loadReportData();
}

async function loadReportData(filters = {}) {
    try {
        showLoading();
        console.log('[DEBUG] Loading report data with filters:', filters);
        
        const queryParams = new URLSearchParams(filters).toString();
        const url = `${API_URLS.reportData}?${queryParams}`;
        console.log('[DEBUG] Fetching from URL:', url);
        
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        reportData = await response.json();
        console.log('[DEBUG] Report data loaded:', reportData);
        
        renderReport();
        showMessage('Report data loaded successfully', 'success');
        
    } catch (error) {
        console.error('[ERROR] Error loading report data:', error);
        showMessage(`Failed to load report data: ${error.message}`, 'error');
        renderEmptyState();
    } finally {
        hideLoading();
    }
}

function renderReport() {
    if (!reportData) {
        console.log('[DEBUG] No report data available');
        renderEmptyState();
        return;
    }
    
    console.log('[DEBUG] Rendering report type:', currentReportType);
    const container = document.getElementById('reportContent');
    
    switch (currentReportType) {
        case 'overview':
            container.innerHTML = renderOverviewReport();
            break;
        case 'performance':
            container.innerHTML = renderPerformanceReport();
            break;
        case 'predictions':
            container.innerHTML = renderPredictionsReport();
            break;
        case 'risk':
            container.innerHTML = renderRiskAnalysisReport();
            break;
        default:
            container.innerHTML = renderOverviewReport();
    }
    
    setTimeout(() => initializeCharts(), 100);
}

function renderOverviewReport() {
    const data = reportData.overview || {};
    
    return `
        <div class="report-section">
            <h2>System Overview</h2>
            <div class="stats-grid">
                <div class="stat-card blue">
                    <div class="label">Total Predictions</div>
                    <div class="value">${(data.total_predictions || 0).toLocaleString()}</div>
                    <div class="description">All time predictions</div>
                </div>
                <div class="stat-card green">
                    <div class="label">Average Likelihood</div>
                    <div class="value">${(data.average_likelihood || 0).toFixed(1)}%</div>
                    <div class="description">Mean pass probability</div>
                </div>
                <div class="stat-card red">
                    <div class="label">At Risk Students</div>
                    <div class="value">${(data.at_risk_students || 0).toLocaleString()}</div>
                    <div class="description">Below 50% likelihood</div>
                </div>
                <div class="stat-card yellow">
                    <div class="label">Active Users</div>
                    <div class="value">${(data.active_users || 0).toLocaleString()}</div>
                    <div class="description">Total registered users</div>
                </div>
            </div>
        </div>
        
        <div class="report-section">
            <h2>Prediction Trends</h2>
            <div class="chart-container">
                <canvas id="trendChart"></canvas>
            </div>
        </div>
        
        <div class="report-section">
            <h2>Risk Distribution</h2>
            <div class="chart-container">
                <canvas id="riskDistributionChart"></canvas>
            </div>
        </div>
        
        <div class="report-section">
            <h2>Key Insights</h2>
            <div class="summary-box">
                <h3>Summary</h3>
                <ul>
                    <li>Total of ${(data.total_predictions || 0).toLocaleString()} predictions made</li>
                    <li>${(data.at_risk_students || 0).toLocaleString()} students identified as at-risk</li>
                    <li>Average pass likelihood is ${(data.average_likelihood || 0).toFixed(1)}%</li>
                    <li>${(data.active_users || 0).toLocaleString()} active users in the system</li>
                </ul>
            </div>
        </div>
    `;
}

function renderPerformanceReport() {
    const data = reportData.performance || {};
    
    return `
        <div class="report-section">
            <h2>Model Performance Metrics</h2>
            <div class="stats-grid">
                <div class="stat-card blue">
                    <div class="label">RMSE</div>
                    <div class="value">${(data.rmse || 0).toFixed(4)}</div>
                    <div class="description">Root Mean Squared Error</div>
                </div>
                <div class="stat-card green">
                    <div class="label">MAE</div>
                    <div class="value">${(data.mae || 0).toFixed(4)}</div>
                    <div class="description">Mean Absolute Error</div>
                </div>
                <div class="stat-card yellow">
                    <div class="label">R² Score</div>
                    <div class="value">${(data.r2_score || 0).toFixed(4)}</div>
                    <div class="description">Coefficient of Determination</div>
                </div>
                <div class="stat-card red">
                    <div class="label">MSE</div>
                    <div class="value">${(data.mse || 0).toFixed(4)}</div>
                    <div class="description">Mean Squared Error</div>
                </div>
            </div>
        </div>
        
        <div class="report-section">
            <h2>Feature Importance</h2>
            <div class="chart-container">
                <canvas id="featureImportanceChart"></canvas>
            </div>
        </div>
        
        <div class="report-section">
            <h2>Model Accuracy Over Time</h2>
            <div class="chart-container">
                <canvas id="accuracyTrendChart"></canvas>
            </div>
        </div>
        
        <div class="report-section">
            <h2>Performance Summary</h2>
            <div class="summary-box">
                <h3>Model Status</h3>
                <ul>
                    <li>Active Model: ${data.model_name || 'N/A'}</li>
                    <li>Model Type: ${data.model_type || 'N/A'}</li>
                    <li>Last Trained: ${data.trained_at || 'N/A'}</li>
                    <li>Total Predictions: ${(data.total_predictions || 0).toLocaleString()}</li>
                </ul>
            </div>
        </div>
    `;
}

function renderPredictionsReport() {
    const data = reportData.predictions || [];
    
    let tableRows = '';
    data.forEach((pred, index) => {
        const riskClass = pred.likelihood >= 70 ? 'low' : pred.likelihood >= 50 ? 'medium' : 'high';
        const riskLabel = pred.likelihood >= 70 ? 'Low Risk' : pred.likelihood >= 50 ? 'Medium Risk' : 'High Risk';
        
        tableRows += `
            <tr>
                <td>${index + 1}</td>
                <td>${pred.student_name || 'N/A'}</td>
                <td>${pred.student_id || 'N/A'}</td>
                <td>${(pred.likelihood || 0).toFixed(1)}%</td>
                <td><span class="risk-badge risk-${riskClass}">${riskLabel}</span></td>
                <td>${pred.prediction_date || 'N/A'}</td>
            </tr>
        `;
    });
    
    return `
        <div class="report-section">
            <h2>Recent Predictions</h2>
            <table class="report-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Student Name</th>
                        <th>Student ID</th>
                        <th>Likelihood</th>
                        <th>Risk Level</th>
                        <th>Date</th>
                    </tr>
                </thead>
                <tbody>
                    ${tableRows || '<tr><td colspan="6" style="text-align: center;">No predictions available</td></tr>'}
                </tbody>
            </table>
        </div>
        
        <div class="report-section">
            <h2>Prediction Distribution</h2>
            <div class="chart-container">
                <canvas id="predictionDistributionChart"></canvas>
            </div>
        </div>
        
        <div class="report-section">
            <h2>Likelihood Range Analysis</h2>
            <div class="chart-container">
                <canvas id="likelihoodRangeChart"></canvas>
            </div>
        </div>
    `;
}

function renderRiskAnalysisReport() {
    const data = reportData.risk_analysis || {};
    
    return `
        <div class="report-section">
            <h2>Risk Analysis Overview</h2>
            <div class="stats-grid">
                <div class="stat-card red">
                    <div class="label">High Risk</div>
                    <div class="value">${(data.high_risk || 0).toLocaleString()}</div>
                    <div class="description">Below 50% likelihood</div>
                </div>
                <div class="stat-card yellow">
                    <div class="label">Medium Risk</div>
                    <div class="value">${(data.medium_risk || 0).toLocaleString()}</div>
                    <div class="description">50-70% likelihood</div>
                </div>
                <div class="stat-card green">
                    <div class="label">Low Risk</div>
                    <div class="value">${(data.low_risk || 0).toLocaleString()}</div>
                    <div class="description">Above 70% likelihood</div>
                </div>
                <div class="stat-card blue">
                    <div class="label">Total Assessed</div>
                    <div class="value">${(data.total_assessed || 0).toLocaleString()}</div>
                    <div class="description">All students</div>
                </div>
            </div>
        </div>
        
        <div class="report-section">
            <h2>Risk Factor Analysis</h2>
            <div class="chart-container">
                <canvas id="riskFactorsChart"></canvas>
            </div>
        </div>
        
        <div class="report-section">
            <h2>Risk Trends Over Time</h2>
            <div class="chart-container">
                <canvas id="riskTrendsChart"></canvas>
            </div>
        </div>
        
        <div class="report-section">
            <h2>High Risk Students</h2>
            <div class="summary-box">
                <h3>Recommendations</h3>
                <ul>
                    <li>Provide additional tutoring for ${(data.high_risk || 0).toLocaleString()} high-risk students</li>
                    <li>Monitor ${(data.medium_risk || 0).toLocaleString()} medium-risk students closely</li>
                    <li>Implement early intervention programs</li>
                    <li>Regular assessment and feedback sessions</li>
                </ul>
            </div>
        </div>
    `;
}

function initializeCharts() {
    destroyCharts();
    console.log('[DEBUG] Initializing charts for:', currentReportType);
    
    switch (currentReportType) {
        case 'overview':
            initOverviewCharts();
            break;
        case 'performance':
            initPerformanceCharts();
            break;
        case 'predictions':
            initPredictionsCharts();
            break;
        case 'risk':
            initRiskCharts();
            break;
    }
}

function initOverviewCharts() {
    const trendCtx = document.getElementById('trendChart');
    if (trendCtx) {
        const trendData = reportData.trend_data || { labels: [], values: [] };
        charts.trendChart = new Chart(trendCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: trendData.labels,
                datasets: [{
                    label: 'Average Likelihood (%)',
                    data: trendData.values,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 5,
                    pointBackgroundColor: '#667eea',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, position: 'top' }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { callback: value => value + '%' }
                    }
                }
            }
        });
    }
    
    const riskCtx = document.getElementById('riskDistributionChart');
    if (riskCtx) {
        const riskData = reportData.risk_distribution || { high: 0, medium: 0, low: 0 };
        charts.riskDistributionChart = new Chart(riskCtx.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: ['High Risk', 'Medium Risk', 'Low Risk'],
                datasets: [{
                    data: [riskData.high, riskData.medium, riskData.low],
                    backgroundColor: ['#ef4444', '#f59e0b', '#10b981'],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });
    }
}

function initPerformanceCharts() {
    const featureCtx = document.getElementById('featureImportanceChart');
    if (featureCtx) {
        const features = reportData.feature_importance || {};
        const sortedFeatures = Object.entries(features)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 10);
        
        charts.featureImportanceChart = new Chart(featureCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: sortedFeatures.map(([name]) => formatFeatureName(name)),
                datasets: [{
                    label: 'Importance',
                    data: sortedFeatures.map(([, value]) => value),
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderColor: '#667eea',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        ticks: { callback: value => (value * 100).toFixed(0) + '%' }
                    }
                }
            }
        });
    }
    
    const accuracyCtx = document.getElementById('accuracyTrendChart');
    if (accuracyCtx) {
        const accuracyData = reportData.accuracy_trend || { labels: [], values: [] };
        charts.accuracyTrendChart = new Chart(accuracyCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: accuracyData.labels,
                datasets: [{
                    label: 'R² Score',
                    data: accuracyData.values,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, max: 1 }
                }
            }
        });
    }
}

function initPredictionsCharts() {
    const distributionCtx = document.getElementById('predictionDistributionChart');
    if (distributionCtx) {
        const ranges = reportData.likelihood_ranges || {};
        charts.predictionDistributionChart = new Chart(distributionCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'],
                datasets: [{
                    label: 'Number of Students',
                    data: [
                        ranges['0-20'] || 0,
                        ranges['20-40'] || 0,
                        ranges['40-60'] || 0,
                        ranges['60-80'] || 0,
                        ranges['80-100'] || 0
                    ],
                    backgroundColor: [
                        '#ef4444',
                        '#f59e0b',
                        '#eab308',
                        '#84cc16',
                        '#10b981'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }
    
    const likelihoodCtx = document.getElementById('likelihoodRangeChart');
    if (likelihoodCtx) {
        const data = reportData.predictions || [];
        const likelihoods = data.map(p => p.likelihood || 0);
        
        charts.likelihoodRangeChart = new Chart(likelihoodCtx.getContext('2d'), {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Likelihood Distribution',
                    data: likelihoods.map((l, i) => ({ x: i, y: l })),
                    backgroundColor: 'rgba(102, 126, 234, 0.6)',
                    pointRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { callback: value => value + '%' }
                    }
                }
            }
        });
    }
}

function initRiskCharts() {
    const factorsCtx = document.getElementById('riskFactorsChart');
    if (factorsCtx) {
        const factors = reportData.risk_factors || {};
        charts.riskFactorsChart = new Chart(factorsCtx.getContext('2d'), {
            type: 'radar',
            data: {
                labels: Object.keys(factors).map(k => formatFeatureName(k)),
                datasets: [{
                    label: 'Risk Impact',
                    data: Object.values(factors),
                    backgroundColor: 'rgba(239, 68, 68, 0.2)',
                    borderColor: '#ef4444',
                    pointBackgroundColor: '#ef4444',
                    pointBorderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }
    
    const trendsCtx = document.getElementById('riskTrendsChart');
    if (trendsCtx) {
        const trends = reportData.risk_trends || { labels: [], high: [], medium: [], low: [] };
        charts.riskTrendsChart = new Chart(trendsCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: trends.labels,
                datasets: [
                    {
                        label: 'High Risk',
                        data: trends.high,
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        fill: true
                    },
                    {
                        label: 'Medium Risk',
                        data: trends.medium,
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        fill: true
                    },
                    {
                        label: 'Low Risk',
                        data: trends.low,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'top' }
                }
            }
        });
    }
}

function destroyCharts() {
    Object.values(charts).forEach(chart => {
        if (chart) chart.destroy();
    });
    charts = {};
}

function renderEmptyState() {
    const container = document.getElementById('reportContent');
    container.innerHTML = `
        <div class="empty-state">
            <h3>No Data Available</h3>
            <p>There is no report data to display. Please check your filters or try again later.</p>
        </div>
    `;
}

async function exportToPDF() {
    try {
        showMessage('Generating PDF report...', 'info');
        
        const filters = {
            reportType: currentReportType,
            dateFrom: document.getElementById('dateFrom')?.value,
            dateTo: document.getElementById('dateTo')?.value
        };
        
        const queryParams = new URLSearchParams(filters).toString();
        const response = await fetch(`${API_URLS.exportPDF}?${queryParams}`);
        
        if (!response.ok) throw new Error('Failed to generate PDF');
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `report_${currentReportType}_${new Date().toISOString().split('T')[0]}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        showMessage('PDF report downloaded successfully', 'success');
        
    } catch (error) {
        console.error('[ERROR] Error exporting PDF:', error);
        showMessage('Failed to export PDF. Please try again.', 'error');
    }
}

async function exportToCSV() {
    try {
        showMessage('Generating CSV export...', 'info');
        
        const filters = {
            reportType: currentReportType,
            dateFrom: document.getElementById('dateFrom')?.value,
            dateTo: document.getElementById('dateTo')?.value
        };
        
        const queryParams = new URLSearchParams(filters).toString();
        const response = await fetch(`${API_URLS.exportCSV}?${queryParams}`);
        
        if (!response.ok) throw new Error('Failed to generate CSV');
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `report_${currentReportType}_${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        showMessage('CSV export downloaded successfully', 'success');
        
    } catch (error) {
        console.error('[ERROR] Error exporting CSV:', error);
        showMessage('Failed to export CSV. Please try again.', 'error');
    }
}

function printReport() {
    window.print();
}

function formatFeatureName(name) {
    return name.replace(/_/g, ' ')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function showLoading() {
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.id = 'loadingOverlay';
    overlay.innerHTML = `
        <div class="loading-spinner">
            <div class="spinner"></div>
            <p>Loading report data...</p>
        </div>
    `;
    document.body.appendChild(overlay);
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.remove();
    }
}

function showMessage(message, type = 'info') {
    const container = document.getElementById('messageContainer');
    if (!container) return;
    
    const messageEl = document.createElement('div');
    messageEl.className = `message message-${type}`;
    messageEl.textContent = message;
    
    container.appendChild(messageEl);
    
    setTimeout(() => {
        messageEl.remove();
    }, 5000);
}