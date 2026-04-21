const fileInput = document.getElementById("fileInput");
const result = document.getElementById("result");
const preview = document.getElementById("preview");
const button = document.getElementById("predictBtn");
const uploadBtn = document.getElementById("uploadBtn");

const API_URL = "https://projectagriatt.onrender.com";

// class names
const names = {
  tomato_early_blight: "Early Blight",
  tomato_late_blight: "Late Blight",
  tomato_healthy: "Healthy Leaf",
};

const validImageTypes = ["image/jpeg", "image/png", "image/jpg"];

// image validation
function isValidType(file) {
  return (
    file &&
    (validImageTypes.includes(file.type) ||
      !!file.name.match(/\.(jpeg|jpg|png)$/i))
  );
}

// upload button trigger
uploadBtn.onclick = () => fileInput.click();

// file selection
fileInput.onchange = () => {
  const file = fileInput.files[0];
  if (!isValidType(file)) {
    alert("Please upload a valid image (jpeg, jpg, png).");
    fileInput.value = "";
    preview.style.display = "none";
    return;
  }
  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
};

// drag and drop
const card = document.querySelector(".card");

card.addEventListener("dragover", (e) => {
  e.preventDefault();
  card.style.border = "2px dashed #22c55e";
});

card.addEventListener("dragleave", () => {
  card.style.border = "";
});

card.addEventListener("drop", (e) => {
  e.preventDefault();
  card.style.border = "";
  const file = e.dataTransfer.files[0];
  if (!isValidType(file)) {
    alert("Please upload a valid image (jpeg, jpg, png).");
    return;
  }
  const dt = new DataTransfer();
  dt.items.add(file);
  fileInput.files = dt.files;
  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
});

button.addEventListener("click", async () => {
  const file = fileInput.files[0];

  if (!file) {
    alert("Please select an image first.");
    return;
  }
  if (!isValidType(file)) {
    alert("Please upload a valid image (jpeg, jpg, png).");
    return;
  }

  // preview
  preview.src = URL.createObjectURL(file);

  const formData = new FormData();
  formData.append("file", file);
  result.innerText = "⏳ Analyzing...";

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error("Server error");

    const data = await response.json();
    const predictions = data.predictions;

    if (!predictions || predictions.length === 0) {
      result.innerText = "⚠️ No predictions returned.";
      return;
    }

    const top1 = predictions[0];
    const label1 = names[top1.label] || top1.label;
    let output = `🧠 ${label1} (${(top1.confidence * 100).toFixed(2)}%)`;

    if (predictions.length >= 2) {
      const top2 = predictions[1];
      const label2 = names[top2.label] || top2.label;
      output += `\n💡 Next best guess: ${label2} (${(top2.confidence * 100).toFixed(2)}%)`;
    }

    if (top1.confidence < 0.4) {
      output += "\n⚠️ Low confidence — try a clearer image.";
    }

    result.innerText = output;
  } catch (err) {
    result.innerText = "❌ Error connecting to server.";
  }
});
