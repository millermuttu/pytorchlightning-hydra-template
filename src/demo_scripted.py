import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import List, Tuple

import torch
import hydra
import gradio as gr
from omegaconf import DictConfig

# from src import utils

# log = utils.get_pylogger(__name__)

classes_names = ['airplane','automobile','bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def demo(cfg: DictConfig) -> Tuple[dict, dict]:
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    # log.info("Running Demo")

    # log.info(f"Instantiating scripted model <{cfg.ckpt_path}>")
    model = torch.jit.load(cfg.ckpt_path)

    # log.info(f"Loaded Model: {model}")

    def image_infer(image):
        if image is None:
            return None
        my_img_tensor = torch.tensor(image, dtype=torch.float32).reshape(1,3,32,32)
        # my_img_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        print("@@@@@@@@@@@@@@@@@@@@@@")
        print(my_img_tensor.dtype, my_img_tensor.shape)
        preds = model.forward_jit(my_img_tensor)
        preds = preds[0].tolist()
        return {classes_names[i]: preds[i] for i in range(10)}

    im = gr.Image(source="upload", type='numpy')

    demo = gr.Interface(
        fn=image_infer,
        inputs=[im],
        outputs=[gr.Label(num_top_classes=10)],
        live=True,
    )

    demo.launch(inbrowser=True)

@hydra.main(
    version_base="1.2", config_path=root / "configs", config_name="demo_scripted.yaml"
)
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()
