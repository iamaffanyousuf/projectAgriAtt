const fileInput = document.getElementById("fileInput");
const result = document.getElementById("result");
const preview = document.getElementById("preview");
const button = document.getElementById("predictBtn");

const names = {
  tomato_early_blight: "Early Blight",
  tomato_late_blight: "Late Blight",
  tomato_healthy: "Healthy Leaf",
};

button.addEventListener("click", async () => {
  const file = fileInput.files[0];

  if (!file) {
    alert("Please select an image!");
    return;
  }

  // Preview
  preview.src = URL.createObjectURL(file);

  const formData = new FormData();
  formData.append("file", file);

  result.innerText = "⏳ Analyzing...";

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    const predictions = data.predictions;

    const top1 = predictions[0];
    const top2 = predictions[1];

    const label1 = names[top1.label] || top1.label;
    const label2 = names[top2.label] || top2.label;

    let output =
      `🧠 ${label1} (${(top1.confidence * 100).toFixed(2)}%)` +
      `\n💡 Next best guess: ${label2} (${(top2.confidence * 100).toFixed(2)}%)`;

    if (top1.confidence < 0.4) {
      output += `\n💡 Low Confidence`;
    }

    result.innerText = output;
  } catch (err) {
    result.innerText = "❌ Error connecting to server";
  }
});
