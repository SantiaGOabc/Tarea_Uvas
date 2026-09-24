import requests
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

MODEL_PATH = "modelo_uvas_final_final.keras"
API_KEY = os.getenv("API_KEY")

# Cargar modelo
model = load_model(MODEL_PATH)

print("Modelo cargado correctamente")
print("Entrada esperada:", model.input_shape)
print("Salida:", model.output_shape)

CLASS_NAMES = [
    "BlackMeasles",
    "BlackRot",
    "HealthyGrapes",
    "LeafBlight"
]


def preprocess_image(image):
    """
    Prepara la imagen para el modelo.
    """
    input_shape = model.input_shape
    height = input_shape[1]
    width = input_shape[2]

    image = image.convert("RGB")
    image = image.resize((width, height))

    image_array = np.array(image).astype("float32")
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No se recibió ninguna imagen"
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "No se seleccionó ninguna imagen"
        }), 400

    try:
        # Abrir imagen
        image = Image.open(file.stream)

        # Preprocesar
        processed_image = preprocess_image(image)

        # Predicción
        prediction = model.predict(processed_image, verbose=0)

        class_index = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][class_index])

        # Nombre de la clase
        if class_index < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_index]
        else:
            class_name = f"Clase {class_index}"

        return jsonify({
            "success": True,
            "class": class_name,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:

        print("Error:", e)

        return jsonify({
            "error": "No se pudo analizar la imagen"
        }), 500

DEEPSEEK_API_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-flash" 

DISEASE_TRANSLATIONS = {
    "BlackRot": "Pudrición negra (Guignardia bidwellii)",
    "BlackMeasles": "Esca o yesca de la vid (Black Measles)",
    "LeafBlight": "Tizón de la hoja (Isariopsis clavispora)",
    "HealthyGrapes": "Hoja sana (sin enfermedades aparentes)"
}

@app.route("/recommendations", methods=["POST"])
def get_recommendations():
    data = request.get_json()
    disease_class = data.get("disease_class")

    if not disease_class:
        return jsonify({"error": "No se proporcionó la enfermedad"}), 400

    # Si la hoja está sana, damos recomendaciones preventivas de mantenimiento
    enfermedad_nombre = DISEASE_TRANSLATIONS.get(disease_class, disease_class)

    prompt = f"""
    Eres un agrónomo y fitopatólogo experto en el cultivo de la vid.
    Un análisis visual ha detectado: "{enfermedad_nombre}" en una hoja de uva.

    Proporciona una guía clara, profesional y concisa con el siguiente formato:
    1. Breve explicación del diagnóstico y sintomas (1-2 oraciones).
    2. Medidas preventivas para evitar su propagación o futuros rebrotes (poda, aireación, riego, fungicidas preventivos).
    3. Acciones inmediatas de tratamiento y curación (métodos culturales y químicos/biológicos si aplica).

    Responde en español de forma directa y fácil de entender para un agricultor o jardinero.
    """

    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "Eres un agrónomo y fitopatólogo experto en el cultivo de la vid. Responde siempre en español de forma directa y fácil de entender."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False
            },
            timeout=180
        )

        if response.status_code != 200:
            return jsonify({"error": "No se pudo comunicar con DeepSeek"}), 500

        result_data = response.json()
        recommendation_text = result_data["choices"][0]["message"]["content"]

        return jsonify({
            "success": True,
            "recommendation": recommendation_text
        })

    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "No hay conexión con DeepSeek. Revisa tu conexión a internet."
        }), 503
    except requests.exceptions.Timeout:
        return jsonify({
            "error": "DeepSeek tardó demasiado en responder. Inténtalo nuevamente."
        }), 504
    except (KeyError, IndexError):
        return jsonify({
            "error": "Respuesta inesperada de DeepSeek. Inténtalo nuevamente."
        }), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=5000
    )