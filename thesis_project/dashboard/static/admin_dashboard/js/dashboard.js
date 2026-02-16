function getCsrfToken() {
    return document.querySelector('[name=csrfmiddlewaretoken]')?.value ||
        document.cookie.match(/csrftoken=([^;]+)/)?.[1] || '';
}

const API_URLS = {
    dashboardStats: '/dashboard/api/dashboard/stats/',
    trendData: '/dashboard/api/dashboard/trends/',
    exportCSV: '/dashboard/api/dashboard/export-csv/',
    exportPDF: '/dashboard/api/dashboard/export-pdf/'
};

let trendChart = null;
let dashboardData = null;
let currentPeriod = 'monthly';

document.addEventListener('DOMContentLoaded', function () {
    loadDashboardData();
    setupEventListeners();
});

function setupEventListeners() {
    document.getElementById('refreshBtn')?.addEventListener('click', loadDashboardData);
    document.getElementById('downloadCsvBtn')?.addEventListener('click', downloadCSV);
    document.getElementById('generatePdfBtn')?.addEventListener('click', generatePDF);
    document.getElementById('trendPeriod')?.addEventListener('change', handlePeriodChange);
}

function handlePeriodChange(event) {
    currentPeriod = event.target.value;
    loadTrendData(currentPeriod);
}

async function loadDashboardData() {
    try {
        showLoading();

        const response = await fetch(API_URLS.dashboardStats);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const data = await response.json();
        dashboardData = data;

        updateKPICards(data);
        updateModelMetrics(data);
        updateFeatureImportance(data);
        updateUserStatistics(data);
        updateModelStatus(data);
        updateLastUpdated();

        await loadTrendData(currentPeriod);

        showMessage('Dashboard data loaded successfully', 'success');

    } catch (error) {
        console.error('Error loading dashboard:', error);
        showMessage('Failed to load dashboard data. Please try again.', 'error');
    } finally {
        hideLoading();
    }
}

async function loadTrendData(period) {
    try {
        const response = await fetch(`${API_URLS.trendData}?period=${period}`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

        const data = await response.json();
        initOrUpdateTrendChart(data);

    } catch (error) {
        console.error('Error loading trend data:', error);

        // Fallback to default data structure
        const defaultData = generateDefaultTrendData(period);
        initOrUpdateTrendChart(defaultData);
    }
}

function generateDefaultTrendData(period) {
    let labels, values;

    if (period === 'weekly') {
        labels = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6', 'Week 7', 'Week 8'];
        values = Array(8).fill(0);
    } else if (period === 'monthly') {
        labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        values = Array(12).fill(0);
    } else if (period === 'yearly') {
        const currentYear = new Date().getFullYear();
        labels = Array.from({ length: 5 }, (_, i) => String(currentYear - 4 + i));
        values = Array(5).fill(0);
    }

    return { labels, values };
}

function updateKPICards(data) {
    const kpi = data.kpi_metrics || {};

    document.getElementById('totalPredictions').textContent =
        (kpi.total_predictions || 0).toLocaleString();

    document.getElementById('avgLikelihood').innerHTML =
        `${(kpi.average_likelihood || 0).toFixed(1)}<span class="text-xl">%</span>`;

    document.getElementById('atRiskStudents').textContent =
        (kpi.at_risk_students || 0).toLocaleString();

    document.getElementById('likelyToPass').textContent =
        (kpi.likely_to_pass || 0).toLocaleString();

    if (kpi.trends) {
        document.getElementById('totalPredictionsTrend').textContent =
            `${kpi.trends.predictions >= 0 ? '↑' : '↓'} ${Math.abs(kpi.trends.predictions || 0)}% from last period`;
        document.getElementById('avgLikelihoodTrend').textContent =
            `${kpi.trends.likelihood >= 0 ? '↑' : '↓'} ${Math.abs(kpi.trends.likelihood || 0)}% from last period`;
        document.getElementById('atRiskTrend').textContent =
            `${kpi.trends.at_risk <= 0 ? '↓' : '↑'} ${Math.abs(kpi.trends.at_risk || 0)}% from last period`;
        document.getElementById('likelyToPassTrend').textContent =
            `${kpi.trends.likely_pass >= 0 ? '↑' : '↓'} ${Math.abs(kpi.trends.likely_pass || 0)}% from last period`;
    }
}

function updateModelMetrics(data) {
    const perf = data.model_performance || {};

    document.getElementById('modelRMSE').textContent =
        perf.rmse ? perf.rmse.toFixed(4) : '--';
    document.getElementById('modelMAE').textContent =
        perf.mae ? perf.mae.toFixed(4) : '--';
    document.getElementById('modelR2').textContent =
        perf.r2_score ? perf.r2_score.toFixed(4) : '--';
    document.getElementById('modelMSE').textContent =
        perf.mse ? perf.mse.toFixed(4) : '--';

    // Show CV metrics if available
    if (perf.cv_rmse !== null && perf.cv_rmse !== undefined) {
        document.getElementById('cvMetrics').classList.remove('hidden');
        document.getElementById('modelCVRMSE').textContent = perf.cv_rmse.toFixed(4);
        document.getElementById('modelCVStd').textContent =
            perf.cv_std ? perf.cv_std.toFixed(4) : '--';
    } else {
        document.getElementById('cvMetrics').classList.add('hidden');
    }
}

function updateFeatureImportance(data) {
    const container = document.getElementById('featureImportanceContainer');
    const features = data.feature_importance || {};

    if (Object.keys(features).length === 0) {
        container.innerHTML = '<p class="text-center text-gray-500">No feature importance data available</p>';
        return;
    }

    const sortedFeatures = Object.entries(features)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 8);

    let html = '<div class="space-y-4">';

    sortedFeatures.forEach(([feature, importance]) => {
        const percentage = (importance * 100).toFixed(1);
        html += `
            <div class="feature-item">
                <div class="feature-header">
                    <div class="feature-name">${formatFeatureName(feature)}</div>
                    <div class="feature-value">${percentage}%</div>
                </div>
                <div class="feature-bar">
                    <div class="feature-fill" style="width: ${percentage}%"></div>
                </div>
            </div>
        `;
    });

    html += '</div>';
    container.innerHTML = html;
}

function updateUserStatistics(data) {
    const stats = data.user_statistics || {};

    document.getElementById('totalStudents').textContent =
        stats.total_students || '--';
    document.getElementById('avgGPA').textContent =
        stats.average_gpa ? stats.average_gpa.toFixed(2) : '--';
    document.getElementById('avgInternship').textContent =
        stats.average_internship_grade ? stats.average_internship_grade.toFixed(2) : '--';
    document.getElementById('avgStudyHours').textContent =
        stats.average_study_hours ? stats.average_study_hours.toFixed(1) + ' hrs' : '--';
    document.getElementById('avgSleepHours').textContent =
        stats.average_sleep_hours ? stats.average_sleep_hours.toFixed(1) + ' hrs' : '--';
    document.getElementById('avgAge').textContent =
        stats.average_age ? stats.average_age.toFixed(1) + ' years' : '--';
    document.getElementById('reviewCenterRate').textContent =
        stats.review_center_rate ? `${stats.review_center_rate}%` : '--';
    document.getElementById('scholarshipRate').textContent =
        stats.scholarship_rate ? `${stats.scholarship_rate}%` : '--';
}

function updateModelStatus(data) {
    const perf = data.model_performance || {};

    const modelName = perf.model_name || 'No Model';
    document.getElementById('activeModel').textContent = formatModelName(modelName);

    const modelType = perf.model_type || 'N/A';
    document.getElementById('modelType').textContent = formatModelType(modelType);

    const statusEl = document.getElementById('modelStatus');
    if (modelName === 'No Model') {
        statusEl.textContent = 'NOT TRAINED';
        statusEl.className = 'status-badge status-error';
    } else {
        statusEl.textContent = 'READY';
        statusEl.className = 'status-badge status-ready';
    }

    const lastTrained = perf.trained_at || 'Not trained yet';
    document.getElementById('lastTrained').textContent =
        lastTrained !== 'Not trained yet' ? formatDate(lastTrained) : lastTrained;
}

function initOrUpdateTrendChart(data) {
    const ctx = document.getElementById('trendChart');
    if (!ctx) return;

    const trendData = data || {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        values: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    };

    if (trendChart) {
        trendChart.data.labels = trendData.labels;
        trendChart.data.datasets[0].data = trendData.values;
        trendChart.update();
    } else {
        trendChart = new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: {
                labels: trendData.labels,
                datasets: [{
                    label: 'Pass Likelihood (%)',
                    data: trendData.values,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true,
                    pointRadius: 4,
                    pointBackgroundColor: '#667eea',
                    pointBorderColor: '#1e293b',
                    pointBorderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#1e293b',
                        borderColor: 'rgba(255, 255, 255, 0.1)',
                        borderWidth: 1,
                        titleColor: '#e2e8f0',
                        bodyColor: '#94a3b8',
                        padding: 12,
                        titleFont: { size: 14 },
                        bodyFont: { size: 13 },
                        callbacks: {
                            label: function (context) {
                                return 'Likelihood: ' + context.parsed.y.toFixed(1) + '%';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { font: { size: 11 }, color: '#94a3b8' }
                    },
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            stepSize: 20,
                            callback: value => value + '%',
                            font: { size: 11 },
                            color: '#94a3b8'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        }
                    }
                }
            }
        });
    }
}

function updateLastUpdated() {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
    });
    document.getElementById('lastUpdated').textContent = `Updated: Today, ${timeStr}`;
}

async function downloadCSV() {
    try {
        const response = await fetch(API_URLS.exportCSV);
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `predictions_${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        showMessage('CSV file downloaded successfully', 'success');
    } catch (error) {
        console.error('Error downloading CSV:', error);
        showMessage('Failed to download CSV file', 'error');
    }
}

async function generatePDF() {
    try {
        showMessage('Generating PDF report...', 'info');
        const response = await fetch(API_URLS.exportPDF);
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `dashboard_report_${new Date().toISOString().split('T')[0]}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        showMessage('PDF report generated successfully', 'success');
    } catch (error) {
        console.error('Error generating PDF:', error);
        showMessage('Failed to generate PDF report', 'error');
    }
}

function formatFeatureName(name) {
    return name.replace(/_/g, ' ')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function formatModelName(name) {
    return name.replace(/_/g, ' ')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
}

function formatModelType(type) {
    if (type === 'ensemble_regression') return 'Ensemble Regression';
    if (type === 'regression') return 'Regression';
    return type.replace(/_/g, ' ')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

function showLoading() {
}

function hideLoading() {
}

function showMessage(message, type = 'info') {
    const container = document.getElementById('messageContainer');
    const messageClass = type === 'error' ? 'error-message' :
        type === 'success' ? 'success-message' : 'info-message';
    const iconClass = type === 'error' ? 'fa-circle-exclamation' :
        type === 'success' ? 'fa-circle-check' : 'fa-circle-info';

    const messageEl = document.createElement('div');
    messageEl.className = 'message ' + messageClass;
    messageEl.innerHTML = `
        <i class="fa-solid ${iconClass} toast-icon"></i>
        <span>${message}</span>
        <button class="toast-close" aria-label="Close">&times;</button>
        <div class="toast-progress"></div>
    `;

    messageEl.querySelector('.toast-close').addEventListener('click', () => {
        dismissToast(messageEl);
    });

    container.appendChild(messageEl);

    setTimeout(() => {
        dismissToast(messageEl);
    }, 4000);
}

function dismissToast(el) {
    if (!el.parentElement) return;
    el.style.animation = 'toastFadeOut 0.3s ease forwards';
    el.addEventListener('animationend', () => el.remove(), { once: true });
}

// Auto-refresh every 60 seconds
setInterval(loadDashboardData, 60000);