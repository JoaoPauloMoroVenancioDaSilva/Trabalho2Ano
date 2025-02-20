import streamlit as st
from PIL import Image
import numpy as np
import matplotlib as mlt


# Funções para conversão e binarização de imagem
def grayscale_average(image):
    """Converte a imagem para tons de cinza usando a média dos canais de cor."""
    return np.mean(image, axis=2).astype(np.uint8)

def grayscale_weighted(image):
    """Converte a imagem para tons de cinza usando uma média ponderada."""
    return (0.2989 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.uint8)

def grayscale_improved(image, darkening_factor=1.0, gamma=1.0):
    """Converte a imagem para tons de cinza com escurecimento e correção de gama."""
    gray_image = grayscale_weighted(image) * darkening_factor
    gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)
    return (255 * ((gray_image / 255) ** gamma)).astype(np.uint8)

def apply_intervals(image, num_intervals, min_val=0, max_val=255):
    """Aplica um threshold intervalado para uma imagem em tons de cinza."""
    gray_image = grayscale_average(image)
    interval_size = (max_val - min_val) // num_intervals
    output_image = np.zeros_like(gray_image)

    # Define os limites de cada intervalo
    for i in range(num_intervals):
        lower_bound = min_val + i * interval_size
        upper_bound = min_val + (i + 1) * interval_size if i < num_intervals - 1 else max_val
        output_image[(gray_image >= lower_bound) & (gray_image < upper_bound)] = upper_bound

    return output_image

# Configurações do aplicativo Streamlit
st.title("Processamento de Imagem")
st.write("Faça upload de uma imagem para aplicar diferentes operações de processamento.")

# Carregar imagem
uploaded_file = st.sidebar.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = np.array(Image.open(uploaded_file))
    st.sidebar.image(uploaded_file, caption='Imagem Original', use_column_width=True)

    # Seleção de operação
    operation = st.sidebar.radio("Escolha a operação",
                                 ["Tons de Cinza (Média)",
                                  "Tons de Cinza (Ponderado)",
                                  "Tons de Cinza (Melhorado)",
                                  "Binarização",
                                  "Threshold Intervalado",
                                  "Divisão de Tonalidade (Intervalos)"])
   
    # Parâmetros adicionais para cada operação
    if operation == "Tons de Cinza (Melhorado)":
        darkening_factor = st.sidebar.slider("Fator de Escurecimento", 0.0, 1.0, 0.5)
        gamma = st.sidebar.slider("Correção de Gama", 0.1, 3.0, 1.0)
   
    elif operation == "Binarização":
        threshold = st.sidebar.slider("Threshold para Binarização", 0, 255, 128)
        color_choice = st.sidebar.selectbox("Aplicar em:", ["Tons de Cinza", "Colorido"])
   
    elif operation == "Threshold Intervalado":
        min_val = st.sidebar.slider("Valor Mínimo", 0, 255, 100)
        max_val = st.sidebar.slider("Valor Máximo", 0, 255, 200)
        num_intervals = 2  # Número de intervalos para binarização
   
    elif operation == "Divisão de Tonalidade (Intervalos)":
        num_intervals = st.sidebar.slider("Número de Intervalos",2,10,5)

    # Processamento da imagem
    if operation == "Tons de Cinza (Média)":
        gray_image = grayscale_average(image)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image)
        ax[0].set_title("Imagem Original")
        ax[0].axis("off")
        ax[1].imshow(gray_image, cmap="gray")
        ax[1].set_title("Tons de Cinza (Média)")
        ax[1].axis("off")
        st.pyplot(fig)
        st.write("Histograma:")
        st.bar_chart(np.histogram(gray_image.flatten(), bins=256)[0])

    elif operation == "Tons de Cinza (Ponderado)":
        gray_image = grayscale_weighted(image)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image)
        ax[0].set_title("Imagem Original")
        ax[0].axis("off")
        ax[1].imshow(gray_image, cmap="gray")
        ax[1].set_title("Tons de Cinza (Ponderado)")
        ax[1].axis("off")
        st.pyplot(fig)
        st.write("Histograma:")
        st.bar_chart(np.histogram(gray_image.flatten(), bins=256)[0])

    elif operation == "Tons de Cinza (Melhorado)":
        gray_image = grayscale_improved(image, darkening_factor, gamma)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image)
        ax[0].set_title("Imagem Original")
        ax[0].axis("off")
        ax[1].imshow(gray_image, cmap="gray")
        ax[1].set_title("Tons de Cinza (Melhorado)")
        ax[1].axis("off")
        st.pyplot(fig)
        st.write("Histograma:")
        st.bar_chart(np.histogram(gray_image.flatten(), bins=256)[0])

    elif operation == "Binarização":
        if color_choice == "Tons de Cinza":
            gray_image = grayscale_average(image)
            binary_image = binarize(gray_image, threshold)
        else:
            binary_image = binarize(image, threshold)
        st.image(binary_image, caption="Imagem Binarizada", use_column_width=True)

    elif operation == "Threshold Intervalado":
        interval_image = apply_intervals(image, num_intervals, min_val, max_val)
        st.image(interval_image, caption="Imagem com Threshold Intervalado", use_column_width=True)

    elif operation == "Divisão de Tonalidade (Intervalos)":
        interval_image = apply_intervals(image, num_intervals)
        st.image(interval_image, caption=f"Imagem com {num_intervals} Intervalos de Tonalidade", use_column_width=True)
