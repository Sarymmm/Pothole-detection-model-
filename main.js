const videoFeed = document.getElementById("videoFeed");
const countEl = document.getElementById("count");
const probEl = document.getElementById("probability");
const gpsEl = document.getElementById("gps");
const pauseBtn = document.getElementById("pauseBtn");

let isPaused = false;

document.getElementById("startBtn").onclick = async () => {
    await fetch("/start");
    isPaused = false;
    pauseBtn.textContent = "Pause";

    videoFeed.src = "";
    setTimeout(() => {
        videoFeed.src = "/video_feed?" + new Date().getTime();
    }, 200);
};

pauseBtn.onclick = async () => {
    const res = await fetch("/pause");
    const data = await res.json();

    isPaused = data.status === "paused";
    pauseBtn.textContent = isPaused ? "Resume" : "Pause";
};

document.getElementById("stopBtn").onclick = async () => {
    await fetch("/stop");
    isPaused = false;
    pauseBtn.textContent = "Pause";
    videoFeed.src = "";

    // Reset displayed stats
    countEl.innerText = "0";
    probEl.innerText = "0.000";
    gpsEl.innerText = "--";
};

function updateStats() {
    fetch("/stats")
        .then((res) => res.json())
        .then((data) => {
            countEl.innerText = data.count;
            probEl.innerText = data.probability.toFixed(3);

            if (data.latitude !== null && data.longitude !== null) {
                gpsEl.innerText =
                    data.latitude.toFixed(5) + ", " + data.longitude.toFixed(5);
            } else {
                gpsEl.innerText = "--";
            }
        })
        .catch((err) => console.error("Stats fetch error:", err));
}

setInterval(updateStats, 1000);