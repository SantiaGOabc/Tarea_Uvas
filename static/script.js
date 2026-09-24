const imageInput = document.getElementById("imageInput");
const selectButton = document.getElementById("selectButton");

const uploadArea = document.getElementById("uploadArea");

const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");

const removeButton = document.getElementById("removeButton");
const analyzeButton = document.getElementById("analyzeButton");

const loading = document.getElementById("loading");

const result = document.getElementById("result");
const resultClass = document.getElementById("resultClass");
const confidenceBadge = document.getElementById("confidenceBadge");
const confidenceText = document.getElementById("confidenceText");
const confidenceBar = document.getElementById("confidenceBar");
const newAnalysisButton = document.getElementById("newAnalysisButton");

const error = document.getElementById("error");
const errorText = document.getElementById("errorText");

// Elementos de la IA (Ollama)
const aiRecommendButton = document.getElementById("aiRecommendButton");
const aiLoading = document.getElementById("aiLoading");
const aiOutput = document.getElementById("aiOutput");
const aiText = document.getElementById("aiText");

let selectedFile = null;
let currentDetectedClass = null;


/* Abrir selector */
selectButton.addEventListener("click", () => {
    imageInput.click();
});


/* Seleccionar imagen */
imageInput.addEventListener("change", () => {
    if (imageInput.files.length > 0) {
        handleFile(imageInput.files[0]);
    }
});


/* Drag & Drop */
uploadArea.addEventListener("dragover", (event) => {
    event.preventDefault();
    uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", (event) => {
    event.preventDefault();
    uploadArea.classList.remove("dragover");

    const files = event.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});


/* Procesar archivo */
function handleFile(file) {
    if (!file.type.startsWith("image/")) {
        showError("El archivo seleccionado no es una imagen.");
        return;
    }

    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (event) => {
        previewImage.src = event.target.result;
        uploadArea.classList.add("hidden");
        previewContainer.classList.remove("hidden");
        result.classList.add("hidden");
        error.classList.add("hidden");
    };

    reader.readAsDataURL(file);
}


/* Analizar imagen con el modelo Keras */
analyzeButton.addEventListener("click", async () => {
    if (!selectedFile) {
        return;
    }

    previewContainer.classList.add("hidden");
    loading.classList.remove("hidden");
    error.classList.add("hidden");

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || "Error al analizar la imagen.");
        }

        showResult(data);

    } catch (err) {
        loading.classList.add("hidden");
        showError(err.message);
    }
});


/* Mostrar resultado del análisis */
function showResult(data) {
    loading.classList.add("hidden");
    result.classList.remove("hidden");

    currentDetectedClass = data.class;
    resultClass.textContent = data.class;

    // Resetear visualización de la IA para este nuevo análisis
    aiRecommendButton.classList.remove("hidden");
    aiLoading.classList.add("hidden");
    aiOutput.classList.add("hidden");
    aiText.textContent = "";

    const confidence = Number(data.confidence);
    confidenceBadge.textContent = `${confidence.toFixed(1)}%`;
    confidenceText.textContent = `${confidence.toFixed(1)}%`;

    setTimeout(() => {
        confidenceBar.style.width = `${confidence}%`;
    }, 100);
}


/* Solicitar recomendaciones a Ollama */
aiRecommendButton.addEventListener("click", async () => {
    if (!currentDetectedClass) return;

    aiRecommendButton.classList.add("hidden");
    aiLoading.classList.remove("hidden");
    aiOutput.classList.add("hidden");

    try {
        const response = await fetch("/recommendations", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ disease_class: currentDetectedClass })
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || "No se pudo generar la recomendación.");
        }

        aiText.innerText = data.recommendation;
        aiOutput.classList.remove("hidden");

    } catch (err) {
        aiRecommendButton.classList.remove("hidden");
        showError(err.message);
    } finally {
        aiLoading.classList.add("hidden");
    }
});


/* Reiniciar / Analizar otra */
removeButton.addEventListener("click", reset);
newAnalysisButton.addEventListener("click", reset);

function reset() {
    selectedFile = null;
    currentDetectedClass = null;

    imageInput.value = "";
    previewImage.src = "";
    confidenceBar.style.width = "0%";

    aiRecommendButton.classList.remove("hidden");
    aiLoading.classList.add("hidden");
    aiOutput.classList.add("hidden");
    aiText.textContent = "";

    previewContainer.classList.add("hidden");
    result.classList.add("hidden");
    loading.classList.add("hidden");
    error.classList.add("hidden");
    uploadArea.classList.remove("hidden");
}


/* Mostrar error general */
function showError(message) {
    errorText.textContent = message;
    error.classList.remove("hidden");
}