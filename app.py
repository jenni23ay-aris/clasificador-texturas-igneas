import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Configuración de la interfaz gráfica
st.set_page_config(
    page_title="Petrología Ígnea | Detección con YOLOv8",
    page_icon="🔬",
    layout="centered"
)

# Encabezado Académico
st.title("🔬 Detección Automatizada de Texturas de Desequilibrio Magmático")
st.markdown("""
**Universidad Central del Ecuador - FIGEMPA** *Carrera de Geología | Cátedra de Petrología Ígnea*

Esta aplicación web utiliza un modelo de aprendizaje supervisado **YOLOv8** para detectar y clasificar texturas de cristales en microfotografías de láminas delgadas (ej. textura sieve, reabsorción, coronas de reacción), optimizando el análisis de procesos dinámicos en sistemas magmáticos.
""")

# Descargo de responsabilidad profesional
st.warning("⚠️ **Nota Académica:** Esta herramienta de visión artificial funciona como un vector de apoyo al análisis petrográfico y **no reemplaza la interpretación, criterios ni validación profesional del geólogo** en el microscopio.")

# Función optimizada para cargar el modelo YOLOv8 una sola vez
@st.cache_resource
def load_yolo_model():
    # Carga el archivo de pesos de la raíz del repositorio
    return YOLO("best.pt")

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Error al cargar el archivo 'best.pt'. Asegúrate de que esté en la raíz del repositorio. Detalle: {e}")
    st.stop()

# Componente para subir la muestra digitalizada
st.subheader("Análisis de Muestra de Lámina Delgada")
uploaded_file = st.file_uploader("Sube una microfotografía (Formatos aceptados: JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Abrir y mostrar la imagen original subida por el usuario
    image = Image.open(uploaded_file)
    st.image(image, caption="Microfotografía original cargada", use_column_width=True)
    
    # Botón para iniciar la inferencia
    if st.button("Ejecutar Detección de Cristales"):
        with st.spinner("Analizando microestructura cristalina..."):
            try:
                # Convertir la imagen PIL a formato adecuado para YOLOv8
                img_input = np.array(image)
                
                # Ejecutar inferencia con YOLOv8
                # El modelo redimensiona internamente la imagen al tamaño de entrenamiento de forma automática
                results = model(img_input)
                
                # Renderizar los resultados en la imagen (cajas, etiquetas y confianzas)
                # results[0].plot() devuelve una matriz en formato BGR (OpenCV)
                res_plotted = results[0].plot()
                
                # Convertir de BGR a RGB para mostrar correctamente en Streamlit
                res_rgb = res_plotted[:, :, ::-1]
                
                st.success("✅ Procesamiento analítico completado")
                st.image(res_rgb, caption="Resultados de la detección automática", use_column_width=True)
                
                # Desplegar información detallada de los objetos detectados si existen
                boxes = results[0].boxes
                if len(boxes) > 0:
                    st.markdown("### 📊 Resumen de Elementos Detectados:")
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0]) * 100
                        st.write(f"• **{class_name}**: Confianza del {confidence:.2f}%")
                else:
                    st.info("No se detectaron texturas claras de desequilibrio magmático en esta sección con el umbral por defecto.")
                    
            except Exception as e:
                st.error(f"Ocurrió un error durante el análisis de la imagen: {e}")