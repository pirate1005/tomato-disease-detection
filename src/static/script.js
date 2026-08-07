const detectionLabel = document.getElementById("detection-label");
const detectionConfidence = document.getElementById("detection-confidence");

const precisionValue = document.getElementById("precision-value");
const recallValue = document.getElementById("recall-value");
const map50Value = document.getElementById("map50-value");
const map5095Value = document.getElementById("map5095-value");


function updateDetection() {
    fetch("/detection_info")
        .then(response => response.json())
        .then(data => {
            detectionLabel.innerText = data.label;
            detectionConfidence.innerText = data.confidence + "%";

            precisionValue.innerText = data.precision;
            recallValue.innerText = data.recall;
            map50Value.innerText = data.map50;
            map5095Value.innerText = data.map5095;
        })
        .catch(error => {
            console.error("Detection error:", error);
        });
}

const saveBtn = document.getElementById("save-btn");

saveBtn.addEventListener("click", () => {

    fetch("/save_detection", {
        method: "POST"
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
    })
    .catch(error => {
        console.error(error);
        alert("Gagal menyimpan data.");
    });

});

setInterval(updateDetection, 1000);