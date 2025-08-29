let isTraining = false;
let currentEpoch = 0;
let totalEpochs = 100;
let currentBatch = 0;
let totalBatches = 100; // UI-only simulation for batch progress
let trainingInterval = null;
let trainingChart = null;

// Metrics storage
let metrics = {
    accuracy: [],
    loss: [],
    precision: [],
    recall: []
};

async function startTraining() {
    if (isTraining) return;

    try {
        isTraining = true;
        currentEpoch = 0;
        currentBatch = 0;

        // Reset metrics and chart
        metrics = { accuracy: [], loss: [], precision: [], recall: [] };
        trainingChart.data.labels = [];
        trainingChart.data.datasets[0].data = [];
        trainingChart.data.datasets[1].data = [];
        trainingChart.update();

        // Update UI status
        document.getElementById('trainingStatus').className = 'status-indicator status-training';
        document.getElementById('statusText').textContent = 'Training';
        document.getElementById('startTraining').style.display = 'none';
        document.getElementById('pauseTraining').style.display = 'inline-block';
        document.getElementById('stopTraining').style.display = 'inline-block';
        addLogEntry('Initiating training session...');

        // Kick off training with defaults; server persists and manages real session
        const response = await fetch('/api/start_training', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ epochs: totalEpochs, batch_size: 32, learning_rate: 0.001 })
        });
        const data = await response.json();

        if (data.status === 'success') {
            addLogEntry(data.message || 'Training started');
            // Ensure totals reflect server config
            if (typeof data.result?.epochs === 'number') {
                totalEpochs = data.result.epochs;
                document.getElementById('totalEpochs').textContent = totalEpochs;
            }
            // Poll server for status
            trainingInterval = setInterval(updateTrainingStatus, 1000);
        } else {
            throw new Error(data.message || 'Failed to start training');
        }
    } catch (error) {
        console.error('Error starting training:', error);
        addLogEntry('Error starting training: ' + error.message);
        stopTraining();
    }
}

function pauseTraining() {
    if (!isTraining) return;
    isTraining = false;
    if (trainingInterval) {
        clearInterval(trainingInterval);
        trainingInterval = null;
    }
    document.getElementById('trainingStatus').className = 'status-indicator status-idle';
    document.getElementById('statusText').textContent = 'Paused';
    document.getElementById('startTraining').style.display = 'inline-block';
    document.getElementById('pauseTraining').style.display = 'none';
    addLogEntry('Training paused');
}

function stopTraining() {
    isTraining = false;
    if (trainingInterval) {
        clearInterval(trainingInterval);
        trainingInterval = null;
    }
    document.getElementById('trainingStatus').className = 'status-indicator status-idle';
    document.getElementById('statusText').textContent = 'Stopped';
    document.getElementById('startTraining').style.display = 'inline-block';
    document.getElementById('pauseTraining').style.display = 'none';
    document.getElementById('stopTraining').style.display = 'none';
    addLogEntry('Training stopped');
}

async function updateTrainingStatus() {
    if (!isTraining) return;
    try {
        const response = await fetch('/api/training_status');
        const data = await response.json();

        if (data.status === 'training') {
            // Update epoch progress
            const epochProgress = Math.max(0, Math.min(100, data.progress || 0));
            document.getElementById('epochProgress').style.width = epochProgress + '%';
            document.getElementById('epochProgress').textContent = Math.round(epochProgress) + '%';

            // Mirror batch progress (no batch granularity from API yet)
            document.getElementById('batchProgress').style.width = epochProgress + '%';
            document.getElementById('batchProgress').textContent = Math.round(epochProgress) + '%';
            document.getElementById('currentBatch').textContent = Math.round((epochProgress / 100) * totalBatches);
            document.getElementById('totalBatches').textContent = totalBatches;

            // Counters
            currentEpoch = data.current_epoch || currentEpoch;
            totalEpochs = data.total_epochs || totalEpochs;
            document.getElementById('currentEpoch').textContent = currentEpoch;
            document.getElementById('totalEpochs').textContent = totalEpochs;

            // Metrics
            if (data.metrics) {
                updateMetrics(
                    Number(data.metrics.accuracy || 0),
                    Number(data.metrics.loss || 0),
                    Number(data.metrics.precision || 0),
                    Number(data.metrics.recall || 0)
                );
            }
        } else if (data.status === 'completed') {
            document.getElementById('epochProgress').style.width = '100%';
            document.getElementById('epochProgress').textContent = '100%';
            document.getElementById('batchProgress').style.width = '100%';
            document.getElementById('batchProgress').textContent = '100%';
            addLogEntry('Training completed successfully!');
            stopTraining();
        } else if (data.status === 'idle') {
            // Keep UI idle if server reports no active session
            document.getElementById('statusText').textContent = 'Idle';
        } else if (data.status === 'error') {
            throw new Error(data.message || 'Training failed');
        }
    } catch (error) {
        console.error('Error updating training status:', error);
        addLogEntry('Error: ' + error.message);
        stopTraining();
    }
}

function updateMetrics(accuracy, loss, precision, recall) {
    // Update metric displays
    document.getElementById('accuracy').textContent = accuracy.toFixed(1) + '%';
    document.getElementById('loss').textContent = loss.toFixed(3);
    document.getElementById('precision').textContent = precision.toFixed(1) + '%';
    document.getElementById('recall').textContent = recall.toFixed(1) + '%';

    // Update chart data with sliding window
    metrics.accuracy.push(accuracy);
    metrics.loss.push(loss);
    if (metrics.accuracy.length > 50) {
        metrics.accuracy.shift();
        metrics.loss.shift();
        trainingChart.data.labels.shift();
    }
    trainingChart.data.labels.push(`Epoch ${currentEpoch}`);
    trainingChart.data.datasets[0].data = metrics.accuracy;
    trainingChart.data.datasets[1].data = metrics.loss;
    trainingChart.update();
}

function initializeChart() {
    const ctx = document.getElementById('trainingChart').getContext('2d');

    trainingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Accuracy',
                    data: [],
                    borderColor: '#28a745',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Loss',
                    data: [],
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { type: 'linear', display: true, position: 'left', max: 100, min: 0 },
                y1: { type: 'linear', display: true, position: 'right', max: 2, min: 0, grid: { drawOnChartArea: false } }
            },
            plugins: { legend: { position: 'top' } }
        }
    });
}

function addLogEntry(message) {
    const logContainer = document.getElementById('trainingLog');
    const timestamp = new Date().toLocaleString();
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    logEntry.innerHTML = `<span class="log-timestamp">[${timestamp}]</span> ${message}`;
    logContainer.appendChild(logEntry);
    logContainer.scrollTop = logContainer.scrollHeight;
}

function downloadModel() {
    addLogEntry('Preparing model download...');
    // Trigger server-side download if a completed model exists
    fetch('/api/training_status')
        .then(r => r.json())
        .then(s => {
            if (s.status === 'completed' && s.model_available) {
                addLogEntry('Model ready. Download starting...');
                window.location.href = '/api/download_model';
            } else {
                addLogEntry('No trained model available yet. Complete training to enable download.');
            }
        })
        .catch(() => {
            // Fallback attempt
            window.location.href = '/api/download_model';
        });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function () {
    initializeChart();
    document.getElementById('trainingStatus').className = 'status-indicator status-idle';
    document.getElementById('statusText').textContent = 'Idle';

    // Wire up buttons by IDs present in template
    const startBtn = document.getElementById('startTraining');
    const stopBtn = document.getElementById('stopTraining');
    const pauseBtn = document.getElementById('pauseTraining');
    if (startBtn) startBtn.addEventListener('click', startTraining);
    if (stopBtn) stopBtn.addEventListener('click', stopTraining);
    if (pauseBtn) pauseBtn.addEventListener('click', pauseTraining);
});
