# import streamlit as st
# import torch
# from PIL import Image
# import sys
# sys.path.insert(0, '.')

# from Pos_Former.lit_posformer import LitPosFormer
# from Pos_Former.datamodule import vocab
# import torchvision.transforms as transforms

# # Cargar modelo una sola vez
# @st.cache_resource
# def load_model():
#     ckp_path = "lightning_logs/version_0/checkpoints/best.ckpt"
#     model = LitPosFormer.load_from_checkpoint(ckp_path, map_location="cpu")
#     model.eval()
#     return model

# # Preprocesar imagen
# def preprocess(image):
#     transform = transforms.Compose([
#         transforms.Grayscale(),
#         transforms.ToTensor(),
#     ])
#     tensor = transform(image).unsqueeze(0)  # [1, 1, h, w]
#     mask = torch.zeros(1, tensor.shape[2], tensor.shape[3], dtype=torch.bool)
#     return tensor, mask

# # Interfaz
# st.title("PosFormer")
# st.subheader("Reconocimiento de Expresiones Matemáticas Manuscritas")
# st.write("Sube una imagen de una expresión matemática y el modelo generará el código LaTeX.")

# uploaded_file = st.file_uploader("Cargar imagen", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Imagen cargada", use_column_width=True)

#     with st.spinner("Reconociendo expresión..."):
#         model = load_model()
#         tensor, mask = preprocess(image)

#         with torch.no_grad():
#             hyps = model.approximate_joint_search(tensor, mask)
#             result = vocab.indices2label(hyps[0].seq)

#     st.success("¡Expresión reconocida!")
#     st.markdown("**Resultado en LaTeX:**")
#     st.code(result, language="latex")
#     st.markdown("**Renderizado:**")
#     st.latex(result)



import streamlit as st
import torch
from PIL import Image
import sys
sys.path.insert(0, '.')

from Pos_Former.lit_posformer import LitPosFormer
from Pos_Former.datamodule import vocab
import torchvision.transforms as transforms

# Cargar modelo una sola vez
@st.cache_resource
def load_model():
    ckp_path = "lightning_logs/version_0/checkpoints/best.ckpt"
    model = LitPosFormer.load_from_checkpoint(ckp_path, map_location="cpu")
    model.eval()
    return model

# Preprocesar imagen
def preprocess(image):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0)  # [1, 1, h, w]

    # Si la imagen tiene fondo blanco, invertirla automáticamente
    if tensor.mean() > 0.5:
        tensor = 1 - tensor

    mask = torch.zeros(1, tensor.shape[2], tensor.shape[3], dtype=torch.bool)
    return tensor, mask

# Interfaz
st.title("PosFormer")
st.subheader("Reconocimiento de Expresiones Matemáticas Manuscritas")
st.write("Sube una imagen de una expresión matemática y el modelo generará el código LaTeX.")

uploaded_file = st.file_uploader("Cargar imagen", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_column_width=True)

    with st.spinner("Reconociendo expresión..."):
        model = load_model()
        tensor, mask = preprocess(image)

        with torch.no_grad():
            hyps = model.approximate_joint_search(tensor, mask)
            result = vocab.indices2label(hyps[0].seq)

    st.success("¡Expresión reconocida!")
    st.markdown("**Resultado en LaTeX:**")
    st.code(result, language="latex")
    st.markdown("**Renderizado:**")
    st.latex(result)

