import gradio as gr
import mlflow.pyfunc
import numpy as np
from jax.image import resize

model = mlflow.pyfunc.load_model(model_uri="models:/cnn_nnx/1")


def prediction(img):
    img_gray_scale = np.sum(img["layers"][0], axis=-1, keepdims=False)
    # print(img["layers"][0].shape)
    resized_image = np.array(
        resize(img_gray_scale, shape=(28, 28), method="nearest"),
        dtype=np.float64,
    )

    # im = Image.fromarray(np.array(resized_image, dtype=np.uint8)).convert("RGB")
    # im.save("drawing.jpeg")

    probabilities = model.predict(resized_image[np.newaxis, ..., np.newaxis] / 256.0)[0]
    indices = np.argpartition(probabilities, -4)[-4:]
    confidences = {str(i): probabilities[i] for i in indices}
    return confidences


brush = gr.Brush(default_size=20, colors=["rgb(255,255,255)"], color_mode="fixed")
dict_value = {
    "composite": None,
    "layers": [255 * np.ones((600, 800), dtype=np.uint8)],
    "background": None,
}
demo = gr.Interface(
    fn=prediction,
    inputs=gr.Sketchpad(
        # value=dict_value,
        interactive=True,
        brush=brush,
        # image_mode="RGB",
        type="numpy",
    ),
    outputs="label",
    live=True,
)

demo.launch()
