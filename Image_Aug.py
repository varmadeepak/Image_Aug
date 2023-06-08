import streamlit as st
from matplotlib import pyplot as plt
import albumentations as A
from streamlit_lottie import st_lottie
from PIL import Image
# from sidebar_utils import handle_uploaded_image_file
plt.rcParams["figure.figsize"] = (10, 7)
import io

import streamlit as st
import PIL.Image
import numpy as np
st.set_page_config(layout="wide", page_title="Image Augmentation Visualizer")
hide_st_style="""
<style>
footer{visibility:hidden;}
</style>"""
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style.css")

st.markdown("""
<div class="container">
        <span class="text1">welcome to</span>
        <span class="text2">Image Augmentation</span>
    </div>""",unsafe_allow_html=True)
st.markdown(" ")
st.markdown(" ")
st.markdown("***")
st.markdown(''' 
This is the part of **Image Augmentation** created in Streamlit. 
**Credit:** App built in `Python` + `Streamlit` by Deepak and Nehansh.
''')

st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)
icol1,icol2=st.columns(2)
with icol1:
   st.markdown(''' 📊⤵ <p class="big-font"> **For**  `DATA PRE-PROCESSING :` [click for Data Pre-Processing](https://data-preprocessing-toolkit-v1.streamlit.app/)</p> ''', unsafe_allow_html=True)
with icol2: 
    st.markdown(''' 📑⤵ <p class="big-font"> **For**  `TEXT EDA :` [click for Text EDA](https://texteda.streamlit.app/)</p> ''', unsafe_allow_html=True)




def plot_original_image(img, additional_information=None):
    st.markdown(
        f"<h4 style='text-align: center; color: black;'>Original</h5>",
        unsafe_allow_html=True,
    )
    st.image(img, use_column_width=True)
    if additional_information:
        st.markdown(additional_information, unsafe_allow_html=True)


def plot_modified_image(img):
    st.markdown(
        f"<h4 style='text-align: center; color: black;'>Augmented image</h5>",
        unsafe_allow_html=True,
    )
    st.image(img, use_column_width=True)


def spacing():
    st.markdown("<br></br>", unsafe_allow_html=True)


def handle_uploaded_image_file(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        img = PIL.Image.open(io.BytesIO(bytes_data))
        return np.array(img), None

    return None, None



def create_pipeline(transformations: list):
    pipeline = []
    for index, transformation in enumerate(transformations):
        if transformation:
            pipeline.append(index_to_transformation(index))

    return pipeline




def spacing():
    st.markdown("<br></br>", unsafe_allow_html=True)


def plot_audio_transformations(original_image, pipeline: A.Compose, additional_information: str = None):
    cols = [1, 2, 1]
    col_1, col_2, col_3 = st.columns(cols)
    with col_2:
        st.markdown(
            f"<h4 style='text-align: center; color: black;'>Original</h5>",
            unsafe_allow_html=True,
        )
        st.image(original_image)
        if additional_information:
            st.markdown(additional_information, unsafe_allow_html=True)

    modified_image = original_image
    for col_index, individual_transformation in enumerate(pipeline.transforms):
        transformation_name = (
            str(type(individual_transformation)).split("'")[1].split(".")[-1]
        )
        modified_image = individual_transformation(image=modified_image)["image"]
        
        col1, col2, col3 = st.columns(cols)

        with col2:
            st.markdown(
                f"<h4 style='text-align: center; color: black;'>{transformation_name} </h5>",
                unsafe_allow_html=True,
            )
            st.image(modified_image)




def index_to_transformation(index: int):

    if index == 0:
        return A.GaussNoise(p=1.0, var_limit=(0.25, 0.5))
    elif index == 1:
        return A.HorizontalFlip(p=1.0)
    elif index == 2:
        return A.VerticalFlip(p=1.0)
    elif index == 3:
        return A.RandomBrightness(p=1.0, limit=(0.5, 1.5))
    elif index == 4:
        return A.AdvancedBlur(p=1.0, blur_limit=3)
    elif index == 5:
        return A.ChannelShuffle(p=1.0)
    elif index == 6:
        return A.ChannelDropout(p=1.0)
    elif index == 7:
        return A.RandomContrast(p=1.0, limit=(0.5, 1.5))


def action(selected_provided_file, transformations):
    # if file_uploader is not None:
    #     img, additional_information = handle_uploaded_image_file(file_uploader)
    if selected_provided_file == "Dog":
            additional_information = 'Image by <a href="https://pixabay.com/users/dm-jones-9527713/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=3582038">Marsha Jones</a> from <a href="https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=3582038">Pixabay</a>'
            img = plt.imread("dog.jpg")
    elif selected_provided_file == "Flower":
            img = plt.imread("flower.jpg")
            additional_information = 'Image by <a href="https://pixabay.com/users/engin_akyurt-3656355/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=3616249">Engin Akyurt</a> from <a href="https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=3616249">Pixabay</a>'
    pipeline = A.Compose(create_pipeline(transformations))
    plot_audio_transformations(img, pipeline, additional_information)


def main():
    placeholder = st.empty()
    placeholder2 = st.empty()
    placeholder.markdown(
        "# Visualize an image augmentation pipeline\n"
        "### To Perform Individual Augmentation Techniques choose them in side bar .\n"
        # "Once you have chosen the augmentation techniques, select or upload an image.\n"
    )
    placeholder2.markdown(
        "To Perform different techniques at once then choose the below checkboxes and Then click Apply")
    st.markdown("Choose the transformations here:")
    
    col1,col2,col3,col4=st.columns(4)
    with col1:
        gaussian_noise = st.checkbox("GaussianNoise")
    with col2:
        horizontal_flip = st.checkbox("HorizontalFlip")
    with col3:
        vertical_flip = st.checkbox("VerticalFlip")
    with col4:
        random_brightness = st.checkbox("RandomBrightness")
    with col1:
        advanced_blur = st.checkbox("AdvancedBlur")
    with col2:
        channel_shuffle = st.checkbox("ChannelShuffle")
    with col3:
        channel_dropout = st.checkbox("ChannelDropout")
    with col4:
        random_contrast = st.checkbox("RandomContrast")

    # st.sidebar.markdown("---")
    # st.sidebar.markdown("(Optional) Upload an image file here:")
    # file_uploader = st.sidebar.file_uploader(label="", type=[".png", ".jpg", ".jpeg"])
    st.markdown("select a sample file here:")
    selected_provided_file = st.selectbox(
        label="", options=["Flower", "Dog"]
    )

    st.markdown("---")
    if st.button("Apply"):
        placeholder.empty()
        placeholder2.empty()
        transformations = [
            gaussian_noise,
            horizontal_flip,
            vertical_flip,
            random_brightness,
            advanced_blur,
            channel_shuffle,
            channel_dropout,
            random_contrast,

        ]
        action(
            selected_provided_file=selected_provided_file,
            transformations=transformations,
        )

if __name__ == "__main__":
    main()
