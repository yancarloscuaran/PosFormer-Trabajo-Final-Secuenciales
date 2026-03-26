import streamlit as st
import torch
from PIL import Image
import sys
sys.path.insert(0, '.')

from Pos_Former.lit_posformer import LitPosFormer
from Pos_Former.datamodule import vocab
import torchvision.transforms as transforms
import base64
from io import BytesIO

# Configuración inicial de la página de la aplicación web con Streamlit
st.set_page_config(
    page_title="PosFormer · Reconocimiento Matemático",
    page_icon="😃",
    layout="wide",
)

# Estilo personalizado para la interfaz de usuario de Streamlit
st.markdown("""
<style>
/* Fondo morado cálido */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(155deg, #f3eefb 0%, #e8ddf5 40%, #ddd0f0 100%);
    min-height: 100vh;
}
[data-testid="stMain"],
[data-testid="stAppViewBlockContainer"] {
    background: transparent !important;
}

/* Configuración de padding y ancho máximo */
[data-testid="stAppViewBlockContainer"] {
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    max-width: 1150px !important;
}

/* Estilo de las tarjetas */
.card {
    background: rgba(255, 252, 255, 0.82);
    border-radius: 16px;
    box-shadow: 0 2px 20px rgba(120, 80, 180, 0.10), 0 1px 4px rgba(120,80,180,0.07);
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.75rem;
}
.card-title {
    font-size: 15px;
    font-weight: 600;
    color: #4a2d72;
    margin: 0 0 0.75rem;
}

/* Estilo de los badges */
.badge {
    display: inline-block;
    background: #ead9f7;
    color: #5c2d8a;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 12px;
    border-radius: 999px;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

/* Estilo del cargador de archivos */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #a07cc5 !important;
    border-radius: 12px !important;
    background: rgba(220, 200, 245, 0.20) !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"] button {
    background-color: #7c4dab !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
}

/* Ajustes de tamaño para renderizado de fórmulas LaTeX */
[data-testid="stMarkdownContainer"] .katex-display {
    font-size: 1.9em !important;
}
[data-testid="stMarkdownContainer"] .katex {
    font-size: 1.7em !important;
}

/* Estilo del código */
[data-testid="stCode"] pre {
    background: #1e0f30 !important;
    color: #e8d5f5 !important;
    border-radius: 10px !important;
    font-size: 13px !important;
}

/* Estilo de las alertas */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-weight: 500 !important;
}

/* Estilo de los expanders */
[data-testid="stExpander"] {
    background: rgba(255, 252, 255, 0.72) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(160, 124, 197, 0.30) !important;
    margin-bottom: 0.75rem;
}

/* Estilo del spinner */
[data-testid="stSpinner"] { color: #7c4dab; }
hr { border-color: rgba(160, 124, 197, 0.25); }

/* Reducir espacio entre elementos de Streamlit */
[data-testid="stVerticalBlock"] > div { margin-bottom: 0 !important; }
.element-container { margin-bottom: 0.4rem !important; }
</style>
""", unsafe_allow_html=True)

# Función para cargar el modelo entrenado desde un checkpoint
@st.cache_resource

def load_model():
    """
    Carga el modelo entrenado desde un archivo de checkpoint.
    El modelo se inicializa en modo evaluación para realizar inferencias.
    """
    ckp_path = "lightning_logs/version_0/checkpoints/best.ckpt"
    model = LitPosFormer.load_from_checkpoint(ckp_path, map_location="cpu")
    model.eval()
    return model

# Función para preprocesar la imagen cargada

def preprocess(image):
    """
    Convierte la imagen cargada a escala de grises y la transforma en un tensor.
    También invierte los colores si el fondo es claro para adaptarse al modelo.
    Devuelve el tensor de la imagen y una máscara inicializada.
    """
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0)
    if tensor.mean() > 0.5:
        tensor = 1 - tensor
    mask = torch.zeros(1, tensor.shape[2], tensor.shape[3], dtype=torch.bool)
    return tensor, mask

# Función para convertir una imagen a formato base64

def image_to_base64(image):
    """
    Convierte una imagen en un string codificado en base64.
    Esto permite mostrar la imagen en la interfaz web de Streamlit.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Configuración del encabezado de la aplicación
header_col, arch_col = st.columns([3, 1])

with header_col:
    st.markdown("""
    <div style="padding: 0.5rem 0 0.25rem;">
        <span class="badge">CROHME · UAO</span>
        <h1 style="font-size:2rem; font-weight:700; color:#3a1f5c;
                   letter-spacing:-0.5px; margin:5px 0 3px;">😃 PosFormer</h1>
        <p style="font-size:14px; color:#7a5499; margin:0;">
            Reconocimiento de expresiones matemáticas manuscritas con Transformers
        </p>
    </div>
    """, unsafe_allow_html=True)

with arch_col:
    with st.expander("🏗️ Ver arquitectura del modelo", expanded=False):
        try:
            # Carga y muestra la imagen de la arquitectura del modelo si está disponible
            arch_img = Image.open("images/arquitectura.png")
            st.image(arch_img, use_column_width=True)
        except FileNotFoundError:
            st.warning("No se encontró images/arquitectura.png")

st.markdown("<hr style='margin: 0.5rem 0 0.75rem;'>", unsafe_allow_html=True)

# Inicialización de variables para la inferencia
image = None
result = None

# Configuración de las columnas para la interfaz de usuario
col_left, col_right = st.columns(2, gap="large")

with col_left:
    # Sección para cargar la imagen de entrada
    st.markdown('<div class="card"><p class="card-title">🖼️ Cargar imagen</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Imagen",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
    st.caption("Formatos: png, jpg, jpeg")
    if uploaded_file is None:
        st.info("Arrastra o selecciona una imagen con una expresión matemática")
    st.markdown('</div>', unsafe_allow_html=True)

# Procesamiento de la imagen cargada
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    with st.spinner("Reconociendo expresión..."):
        # Carga el modelo y realiza la inferencia sobre la imagen procesada
        model = load_model()
        tensor, mask = preprocess(image)
        with torch.no_grad():
            hyps = model.approximate_joint_search(tensor, mask)
            result = vocab.indices2label(hyps[0].seq)

# Renderizado del resultado en formato LaTeX
with col_left:
    st.markdown('<div class="card"><p class="card-title">📊 Renderizado LaTeX</p>', unsafe_allow_html=True)
    if result is not None:
        st.latex(result)  # Muestra el resultado en formato LaTeX
    else:
        st.markdown(
            '<div style="height:85px;display:flex;align-items:center;'
            'justify-content:center;color:#a07cc5;font-size:13px;">'
            "El renderizado aparecerá aquí</div>",
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

# Vista previa de la imagen cargada y su código LaTeX
with col_right:
    st.markdown('<div class="card"><p class="card-title">🖼️ Vista previa</p>', unsafe_allow_html=True)
    if image is not None:
        # Muestra la imagen cargada en la interfaz
        st.markdown(
            '<div style="display: flex; justify-content: center;">'  # Centra la imagen
            f'<img src="data:image/png;base64,{image_to_base64(image)}" alt="{uploaded_file.name}" width="450">'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)  # Agrega un salto de línea después de la imagen
        st.success("✔  Expresión reconocida")
        st.markdown(
            '<p style="font-size:11px;color:#7a5499;text-transform:uppercase;'
            'letter-spacing:0.07em;margin:0.5rem 0 3px;">Código LaTeX</p>',
            unsafe_allow_html=True,
        )
        st.code(result, language="latex")  # Muestra el código LaTeX generado
    else:
        st.markdown(
            '<div style="height:190px;display:flex;align-items:center;'
            'justify-content:center;color:#a07cc5;font-size:13px;'
            'border:1.5px dashed #a07cc5;border-radius:10px;">'
            "La vista previa aparecerá aquí</div>",
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

# Pie de página con información del proyecto
st.markdown(
    '<p style="text-align:center;font-size:12px;color:#a07cc5;margin-top:1rem;">'
    "PosFormer · Maestría en IA y Ciencia de Datos · UAO</p>",
    unsafe_allow_html=True,
)