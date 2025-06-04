async function predict() {
  const player = document.getElementById("player").value;
  const stat = document.getElementById("stat").value;
  const threshold = document.getElementById("threshold").value;
  const resultDiv = document.getElementById("result");

  resultDiv.innerHTML = "⏳ Loading...";
  try {
    const res = await fetch(`http://127.0.0.1:8000/predict?player=${encodeURIComponent(player)}&stat=${stat}&threshold=${threshold}`);
    const data = await res.json();

    if (data.error) {
      resultDiv.innerHTML = `Error: ${data.error}`;
    } else {
      resultDiv.innerHTML = `
        <h3>Prediction</h3>
        <p><strong>${data.player}</strong> will score <strong>${stat}</strong> over ${data.threshold}?</p>
        <p><strong>${data.prediction}</strong> (${data.confidence})</p>
      `;
    }
  } catch (err) {
    resultDiv.innerHTML = `Failed to fetch prediction: ${err}`;
  }
}
