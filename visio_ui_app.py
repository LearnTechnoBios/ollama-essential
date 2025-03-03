import streamlit as st
from PIL import Image
import ollama
import base64
import io


def image_to_base64(img):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    img.save(buffered, format='JPEG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')



def get_llava_response(image_base64, prompt):
    try:
        response = ollama.chat(
            model="llava:7b",  # Or a suitable alternative
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_base64],
                }
            ]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error: {e}"


def main():
    st.title("Simple Image Analyzer")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image_base64 = image_to_base64(image) # Convert to base64

        prompt = st.text_input("Ask a question about the image:", "Describe the objects in the image")

        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                llm_response = get_llava_response(image_base64, prompt)  # Pass base64 image

            st.subheader("LLM Response:")
            st.write(llm_response)


if __name__ == "__main__":
    main()

