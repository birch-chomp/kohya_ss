import gradio as gr
from typing import Tuple
from .common_gui import (
    get_file_path,
    get_folder_path,
    get_any_file_path,
    document_symbol,
    folder_symbol,
)


class animaTraining:
    def __init__(
        self,
        headless: bool = False,
        finetuning: bool = False,
        training_type: str = "",
        config: dict = {},
        anima_checkbox: gr.Checkbox = False,
    ) -> None:
        self.headless = headless
        self.finetuning = finetuning
        self.training_type = training_type
        self.config = config
        self.anima_checkbox = anima_checkbox

        with gr.Accordion(
            "Anima", open=True, visible=False, elem_classes=["flux1_background"]
        ) as anima_accordion:
            with gr.Group():
                qwen3_ext = gr.Textbox(value="*.safetensors *.ckpt", visible=False)
                qwen3_ext_name = gr.Textbox(value="Model types", visible=False)

                with gr.Row():
                    self.qwen3 = gr.Textbox(
                        label="Qwen3 Text Encoder",
                        placeholder="Path to Qwen3 Text Encoder model",
                        value=self.config.get("anima.qwen3", ""),
                        interactive=True,
                    )
                    self.qwen3_file = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        elem_classes=["tool"],
                        visible=(not headless),
                    )
                    self.qwen3_file.click(
                        get_file_path,
                        inputs=[self.qwen3, qwen3_ext, qwen3_ext_name],
                        outputs=self.qwen3,
                        show_progress=False,
                    )
                    self.qwen3_folder = gr.Button(
                        folder_symbol,
                        elem_id="open_folder_small",
                        elem_classes=["tool"],
                        visible=(not headless),
                    )
                    self.qwen3_folder.click(
                        get_folder_path,
                        inputs=self.qwen3,
                        outputs=self.qwen3,
                        show_progress=False,
                    )

                    self.llm_adapter_path = gr.Textbox(
                        label="LLM Adapter Path",
                        placeholder="Path to LLM Adapter model",
                        value=self.config.get("anima.llm_adapter_path", ""),
                        interactive=True,
                    )
                    self.llm_adapter_path_file = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        elem_classes=["tool"],
                        visible=(not headless),
                    )
                    self.llm_adapter_path_file.click(
                        get_any_file_path,
                        inputs=[self.llm_adapter_path],
                        outputs=self.llm_adapter_path,
                        show_progress=False,
                    )

                    self.t5_tokenizer_path = gr.Textbox(
                        label="T5 Tokenizer Path",
                        placeholder="Path to T5 Tokenizer",
                        value=self.config.get("anima.t5_tokenizer_path", ""),
                        interactive=True,
                    )
                    self.t5_tokenizer_path_folder = gr.Button(
                        folder_symbol,
                        elem_id="open_folder_small",
                        elem_classes=["tool"],
                        visible=(not headless),
                    )
                    self.t5_tokenizer_path_folder.click(
                        get_folder_path,
                        inputs=self.t5_tokenizer_path,
                        outputs=self.t5_tokenizer_path,
                        show_progress=False,
                    )

                with gr.Row():

                    gr.Markdown(
                            """
                            ### Component Learning Rates
                            Leave empty to use base learning rate. Set to 0 to disable the component.  
                            Style + poses: Train Self Attn, MLP  
                            Style only: Train MLP  
                            Character: Train Cross Attn, MLP  
                            Slider/Concept: Train Cross Attn  
                            """)
                    
                with gr.Row():
                    
                    self.self_attn_lr = gr.Number(
                        label="Self Attention",
                        value=self.config.get("anima.self_attn_lr", lambda: None),
                        minimum=-1,
                        maximum=1,
                    )

                    self.cross_attn_lr = gr.Number(
                        label="Cross Attention",
                        value=self.config.get("anima.cross_attn_lr", lambda: None),
                        minimum=-1,
                        maximum=1,
                    )

                    self.mlp_lr = gr.Number(
                        label="MLP",
                        value=self.config.get("anima.mlp_lr", lambda: None),
                        minimum=-1,
                        maximum=1,
                    )

                    self.llm_adapter_lr = gr.Number(
                        label="LLM Adapter",
                        value=self.config.get("anima.llm_adapter_lr", lambda: None),
                        minimum=-1,
                        maximum=1,
                    )

                with gr.Row():

                    self.timestep_sampling = gr.Dropdown(
                        label="Timestep Sampling",
                        choices=["flux_shift", "sigma", "shift", "sigmoid", "uniform"],
                        value=self.config.get("anima.timestep_sampling", "sigma"),
                        interactive=True,
                    )

                    self.discrete_flow_shift = gr.Number(
                        label="Discrete Flow Shift",
                        value=self.config.get("anima.discrete_flow_shift", 3.0),
                        info="Discrete flow shift for the Euler Discrete Scheduler. Default 3.0. This value is used when --timestep_sampling is set to shift.",
                        minimum=-1024,
                        maximum=1024,
                        step=0.01,
                        interactive=True,
                    )

                    self.sigmoid_scale = gr.Number(
                        label="Sigmoid Scale",
                        value=self.config.get("anima.sigmoid_scale", 1.0),
                        info="Scale factor when --timestep_sampling is set to sigmoid, shift, or flux_shift. Default 1.0.",
                        minimum=-1024,
                        maximum=1024,
                        step=0.01,
                        interactive=True,
                    )
                with gr.Row():

                    self.attn_mode = gr.Dropdown(
                        label="Attention Mode",
                        choices=["torch", "xformers", "flash", "sageattn"],
                        value=self.config.get("anima.attn_mode", "torch"),
                        info="This option overrides --xformers.",
                        interactive=True,
                    )

                    self.split_attn = gr.Checkbox(
                        label="Split Attention",
                        value=self.config.get("anima.split_attn", False),
                        info="Split attention computation to reduce memory usage. Required when using --attn_mode xformers.",
                        interactive=True,
                    )

                    # self.attn_mode.change(
                    #     lambda attn_mode, split_attn: gr.Checkbox(
                    #         value=True if attn_mode == "xformers" else split_attn,
                    #         interactive=attn_mode != "xformers"
                    #     ),
                    #     inputs=[self.attn_mode, self.split_attn],
                    #     outputs=[self.split_attn],
                    # )

                with gr.Row(visible=True if not finetuning else False):
                    self.cpu_offload_checkpointing = gr.Checkbox(
                        label="CPU Offload Checkpointing",
                        value=self.config.get("anima.cpu_offload_checkpointing", False),
                        info="[Experimental] Enable offloading of tensors to CPU during checkpointing",
                        interactive=True,
                    )

                    self.unsloth_offload_checkpointing = gr.Checkbox(
                        label="Unsloth Offload Checkpointing",
                        value=self.config.get("anima.unsloth_offload_checkpointing", False),
                        info="Offload activations to CPU RAM using async non-blocking transfers (faster than --cpu_offload_checkpointing). Cannot be combined with --cpu_offload_checkpointing.",
                        interactive=True,
                    )

                with gr.Row():
                    self.qwen3_max_token_length = gr.Number(
                        label="Qwen3 Max Token Length",
                        value=self.config.get("anima.qwen3_max_token_length", 512),
                        info="Max token length for Qwen3",
                        minimum=0,
                        maximum=4096,
                        step=1,
                        interactive=True,
                    )
                    self.t5_max_token_length = gr.Number(
                        label="T5 Max Token Length",
                        value=self.config.get("anima.t5_max_token_length", 512),
                        info="Max token length for T5",
                        minimum=0,
                        maximum=4096,
                        step=1,
                        interactive=True,
                    )

                with gr.Row():
                    self.anima_cache_text_encoder_outputs = gr.Checkbox(
                        label="Cache Text Encoder Outputs",
                        value=self.config.get(
                            "anima.cache_text_encoder_outputs", False
                        ),
                        info="Cache text encoder outputs to speed up inference",
                        interactive=True,
                    )
                    self.anima_cache_text_encoder_outputs_to_disk = gr.Checkbox(
                        label="Cache Text Encoder Outputs to Disk",
                        value=self.config.get(
                            "anima.cache_text_encoder_outputs_to_disk", False
                        ),
                        info="Cache text encoder outputs to disk to speed up inference",
                        interactive=True,
                    )

                self.anima_checkbox.change(
                    lambda anima_checkbox: gr.Accordion(visible=anima_checkbox),
                    inputs=[self.anima_checkbox],
                    outputs=[anima_accordion],
                )