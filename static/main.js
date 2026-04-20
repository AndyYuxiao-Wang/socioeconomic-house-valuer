document.getElementById("predictBtn").addEventListener("click", function() {
    let data = {
        "Postcode": document.getElementById("postcode").value,
        "Type": document.getElementById("ptype").value,
        "Floor Area": document.getElementById("floor_area").value
    };

    fetch("/predict", {
        method: "POST",
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(res => {
        document.getElementById("pred_price").innerText = res.predicted_price.toLocaleString();
        document.getElementById("range_low").innerText = res.recommended_range[0].toLocaleString();
        document.getElementById("range_high").innerText = res.recommended_range[1].toLocaleString();


        map.setView([51.505, -0.09], 10);
    });
});

// Initialize Leaflet map
var map = L.map('map').setView([51.505, -0.09], 6);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19
}).addTo(map);

async function predictPrice() {
    const lsoa = document.getElementById('lsoa').value.trim();
    const type = document.getElementById('type').value;
    const floor = parseFloat(document.getElementById('floor').value);

    if (!lsoa || !type || isNaN(floor)) {
        alert("Please fill in all fields correctly.");
        return;
    }

    const response = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ LSOA21CD: lsoa, Type: type, "Floor Area": floor })
    });

    const data = await response.json();
    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = `
        <strong>Predicted Price:</strong> £${data.predicted_price.toLocaleString()} <br>
        <strong>Recommended Range:</strong> £${data.recommended_range[0].toLocaleString()} - £${data.recommended_range[1].toLocaleString()} <br>
        <strong>Socioeconomic Info:</strong><br>
        Income: £${data.socio.Income.toLocaleString()} <br>
        Education: ${data.socio.Education} <br>
        Crime Decile: ${data.socio.Crime}
    `;

    // Update map
    if (data.lat && data.lon) {
        map.setView([data.lat, data.lon], 14);

        // Add marker with popup
        L.marker([data.lat, data.lon]).addTo(map)
            .bindPopup(resultDiv.innerHTML)
            .openPopup();
    }
}