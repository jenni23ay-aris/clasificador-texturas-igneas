# 🔬 Detección de Texturas de Desequilibrio Magmático con YOLOv8

**Universidad Central del Ecuador** **Facultad de Ingeniería en Geología, Minas, Petróleos y Ambiental (FIGEMPA)** *Carrera de Geología - Quinto Semestre*

## 🎯 Descripción del Proyecto
Este proyecto de investigación semestral aplica técnicas de aprendizaje supervisado para automatizar la identificación de texturas cristalinas en rocas ígneas extrusivas. Utilizando la arquitectura de detección de objetos **YOLOv8**, el sistema localiza y clasifica microestructuras clave como la textura sieve (en cedazo), coronas de reacción y fenómenos de reabsorción en microfotografías de láminas delgadas, proporcionando un indicador cuantitativo de procesos de desequilibrio magmático.

## 🛠️ Herramientas Utilizadas
- **Roboflow:** Gestión de la base de datos petrográfica, aumentación de datos y etiquetado de cristales.
- **Google Colab:** Entorno de computación en la nube utilizado para el entrenamiento del modelo sobre GPU.
- **Ultralytics (YOLOv8):** Framework de visión artificial seleccionado por su alta precisión en localización microestructural.
- **Streamlit & GitHub:** Desarrollo del entorno interactivo web y despliegue en la nube (Streamlit Community Cloud).

## 📁 Arquitectura del Repositorio
- `app.py`: Código principal de la aplicación web interconectado con la interfaz de Streamlit.
- `requirements.txt`: Declaración de dependencias del entorno de ejecución.
- `best.pt`: Parámetros y pesos óptimos resultantes del entrenamiento del modelo YOLOv8.
- `README.md`: Documentación técnica y académica del proyecto.

## 🚀 Guía de Operación
1. Ingrese al enlace de la aplicación desplegada en Streamlit Cloud.
2. Cargue una microfotografía digitalizada de una lámina delgada en formato de imagen estándar (`.png`, `.jpg`).
3. Presione el botón **"Ejecutar Detección de Cristales"**.
4. El sistema devolverá la imagen analizada con cuadros delimitadores sobre los minerales detectados junto a su porcentaje de certidumbre analítica.

⚠️ **Cláusula de Interpretación Petrográfica:** Los resultados generados por este software constituyen un recurso de apoyo computacional. Toda determinación formal debe ser contrastada y corroborada bajo microscopio petrográfico por un profesional de las Ciencias de la Tierra.