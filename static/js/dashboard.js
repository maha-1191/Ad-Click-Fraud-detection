let fraudChartInstance = null;
let timeTrendChartInstance = null;

/* ================= PIE CHART ================= */
function renderFraudChart(fraud, legit) {
    const canvas = document.getElementById("fraudChart");
    if (!canvas || typeof Chart === "undefined") return;

    if (fraudChartInstance) {
        fraudChartInstance.destroy();
    }

    if (fraud === 0 && legit === 0) {
        console.warn("No click data available for pie chart");
        return;
    }

    fraudChartInstance = new Chart(canvas, {
        type: "pie",
        data: {
            labels: ["Fraud Clicks", "Legit Clicks"],
            datasets: [{
                data: [fraud, legit],
                backgroundColor: [
                    "#ef4444",  // red-500 (fraud)
                    "#22c55e"   // green-500 (legit)
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: {
                        boxWidth: 14,
                        padding: 16
                    }
                },
                tooltip: {
                    callbacks: {
                        label: ctx => `${ctx.label}: ${ctx.raw}`
                    }
                }
            }
        }
    });
}

/* ================= FRAUD CLICKS vs HOUR ================= */
function renderFraudClicksChart(labels, fraudClicks) {
    const ctx = document.getElementById("timeTrendChart");
    if (!ctx || typeof Chart === "undefined") return;

    if (timeTrendChartInstance) {
        timeTrendChartInstance.destroy();
    }

    timeTrendChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: "Fraud Clicks",
                data: fraudClicks,
                borderColor: "#ef4444",
                backgroundColor: "rgba(239, 68, 68, 0.12)",
                borderWidth: 3,
                tension: 0.4,
                pointRadius: 4,
                pointBackgroundColor: "#ef4444",
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "top"
                },
                tooltip: {
                    callbacks: {
                        label: ctx => `Fraud Clicks: ${ctx.raw}`
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: "#e5e7eb"
                    },
                    title: {
                        display: true,
                        text: "Hour of Day (0–23)"
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: "#e5e7eb"
                    },
                    title: {
                        display: true,
                        text: "Number of Fraud Clicks"
                    }
                }
            }
        }
    });
}









