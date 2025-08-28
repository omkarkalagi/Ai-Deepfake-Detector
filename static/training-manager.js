let isTraining = false;
let currentEpoch = 0;
let totalEpochs = 100;
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
    if (isTraining) {
        return;
    }
    
    try {
        isTraining = true;
        currentEpoch = 1;
        
        // Reset metrics
        metrics = {
            accuracy: [],
            loss: [],
            precision: [],
            recall: []
        };
        
        // Reset chart
        trainingChart.data.labels = [];
        trainingChart.data.datasets[0].data = [];
        trainingChart.data.datasets[1].data = [];
        trainingChart.update();
        
        // Update status indicator
        document.getElementById('trainingStatus').className = 'status-indicator status-active';
        addLogEntry('Initiating training session...');
        
        // Make API call to start training
        const response = await fetch('/api/start_training', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                epochs: totalEpochs,
                batch_size: 32,
                learning_rate: 0.001
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            addLogEntry(data.message);
            // Start polling for updates
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

function stopTraining() {
    isTraining = false;
    if (trainingInterval) {
        clearInterval(trainingInterval);
        trainingInterval = null;
    }
    document.getElementById('trainingStatus').className = 'status-indicator status-idle';
    addLogEntry('Training stopped');
}

async function updateTrainingStatus() {
    try {
        const response = await fetch('/api/training_status');
        const data = await response.json();
        
        if (data.status === 'training') {
            // Update progress
            const epochProgress = data.progress || 0;
            document.getElementById('epochProgress').style.width = epochProgress + '%';
            document.getElementById('epochProgress').textContent = Math.round(epochProgress) + '%';
            
            // Update epoch counter
            document.getElementById('currentEpoch').textContent = data.current_epoch;
            document.getElementById('totalEpochs').textContent = data.total_epochs;
            
            // Update metrics if available
            if (data.metrics) {
                updateMetrics(
                    data.metrics.accuracy,
                    data.metrics.loss,
                    data.metrics.precision,
                    data.metrics.recall
                );
            }
            
            // Add log entry for significant changes
            if (data.current_epoch > currentEpoch) {
                addLogEntry(`Epoch ${data.current_epoch}/${data.total_epochs} completed`);
                currentEpoch = data.current_epoch;
            }
        } else if (data.status === 'completed') {
            addLogEntry('Training completed successfully!');
            stopTraining();
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
    
    // Update chart data
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
            datasets: [{
                label: 'Accuracy',
                data: [],
                borderColor: '#28a745',
                backgroundColor: 'rgba(40, 167, 69, 0.1)',
                tension: 0.4
            }, {
                label: 'Loss',
                data: [],
                borderColor: '#dc3545',
                backgroundColor: 'rgba(220, 53, 69, 0.1)',
                tension: 0.4,
                yAxisID: 'y1'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    max: 100,
                    min: 0
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    max: 2,
                    min: 0,
                    grid: {
                        drawOnChartArea: false,
                    },
                }
            },
            plugins: {
                legend: {
                    position: 'top'
                }
            }
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

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeChart();
    document.getElementById('trainingStatus').className = 'status-indicator status-idle';
    
    // Add event listeners
    document.querySelector('.btn-start').addEventListener('click', startTraining);
    document.querySelector('.btn-stop').addEventListener('click', stopTraining);
});
