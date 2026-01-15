function renderUsageChart(canvasId, labels, values) {
  const ctx = document.getElementById(canvasId);

  if (!ctx) return;

  // Destroy old chart if already exists
  if (window.usageChartInstance) {
    window.usageChartInstance.destroy();
  }

  window.usageChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "kWh Usage",
          data: values,
          borderWidth: 2,
          tension: 0.3,
          pointRadius: 2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          ticks: { maxTicksLimit: 10 }
        },
        y: {
          beginAtZero: true
        }
      }
    }
  });
}
