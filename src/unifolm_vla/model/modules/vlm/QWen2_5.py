import torch
import torch.nn as nn
from typing import Optional, List
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Dict, Optional, List
from qwen_vl_utils import process_vision_info
from accelerate.logging import get_logger
logger = get_logger(__name__)

IGNORE_INDEX = -100


class _QWen_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen2.5-VL (Qwen2_5_VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    Notes:
        - Keeps original model behavior; does not modify internal architecture.
        - Mixed precision handled via torch.autocast in forward / generate.
        - Adaptation layer can be extended for future multi-modal routing if needed.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen2.5-VL wrapper.

        Parameters:
            config (dict | Any | None):
                Expected to expose a nested attribute/namespace `framework.get("qwenvl", {})`
                where:
                    framework.qwenvl.base_vlm (str): HuggingFace model id or local path.
                Optional expected structure (illustrative):
                    config.framework.get("qwenvl", {}) -> {
                        "base_vlm": "Qwen/Qwen2.5-VL-3B-Instruct"
                    }
                    config.datasets.vla_data.get("CoT_prompt", str) may be used later in build_qwenvl_inputs.
            **kwargs:
                Ignored currently; placeholder for future extension (e.g., override device_map, dtype).

        Side Effects:
            - Downloads / loads pretrained Qwen2.5-VL weights (unless cached).
            - Instantiates AutoProcessor and enforces left padding (required for some FlashAttention paths).

        Attributes Set:
            self.model (Qwen2_5_VLForConditionalGeneration)
            self.processor (AutoProcessor)
            self.config (original config reference)

        Notes:
            - device_map='cuda' is passed to from_pretrained (single or multi-GPU depending on HF accelerate mapping).
            - torch_dtype='auto' lets HF decide best available (prefers bfloat16 on supported hardware).
            - tokenizer padding_side forced to 'left' (important for generation + KV caching alignment).
        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen2.5-VL-7B-Instruct")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"
        
        self.model = model
        self.processor = processor
        self.config = config

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen2.5-VL backbone.

        Args:
            input_ids (LongTensor | None): [B, T] token ids (mutually exclusive with inputs_embeds).
            attention_mask (Tensor | None): [B, T], 1 = attend, 0 = masked.
            pixel_values (FloatTensor | None): Vision batch (model-specific preprocessed shape).
            labels (LongTensor | None): [B, T] LM targets; ignored positions = -100 (IGNORE_INDEX).
            image_grid_thw (FloatTensor | None): Optional tiling metadata (e.g., [B, 3] for temporal/height/width splits).
            inputs_embeds (FloatTensor | None): [B, T, D] alternative embedding input.
            past_key_values (List[FloatTensor] | None): Cached KV states for incremental decoding.
            use_cache (bool | None): If True, returns updated past_key_values.
            output_attentions (bool): Whether to include attention maps.
            output_hidden_states (bool): Must be True if downstream modules consume hidden states.
            return_dict (bool): Return HF dataclass if True; else tuple.
            **kwargs: Extra args forwarded to underlying model.

        Returns:
            CausalLMOutputWithPast | tuple: HF-standard structure (logits, past_key_values, hidden_states, etc.).

        Notes:
            - Autocast(bfloat16) used for efficiency.
            - padding_side already set to 'left' in tokenizer at init.
            - Hidden states required for auxiliary alignment or feature extraction modules.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        return outputs

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.FloatTensor = None,
        max_new_tokens: int = 128,
        output_hidden_states: bool = True,
        return_dict_in_generate: bool = True,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            input_ids (LongTensor): [B, T] prompt tokens.
            attention_mask (Tensor | None): [B, T] mask (0 = pad).
            pixel_values (FloatTensor | None): Optional vision inputs aligned with prompts.
            max_new_tokens (int): Maximum number of new tokens to sample/generate.
            output_hidden_states (bool): Whether to keep hidden states during generation.
            return_dict_in_generate (bool): Return structured GenerateOutput if True.
            **kwargs: Passed to model.generate (e.g., temperature, top_p, do_sample, eos_token_id, repetition_penalty).

        Returns:
            GenerateOutput | Model-dependent generation return.

        Notes:
            - Uses autocast(float16); relies on attribute enable_mixed_precision_training.
            - For iterative dialogue, caller manages past_key_values externally.
        """
        with torch.autocast("cuda", enabled=self.enable_mixed_precision_training, dtype=torch.float16):
            generation_output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, **kwargs):
        """
        Construct and tokenize multimodal chat-style inputs for Qwen2.5-VL (batched).

        Overview:
            For each sample i:
                - Takes a list of PIL images: images[i] = [img_0, img_1, ...]
                - Takes a matching instruction string instructions[i]
                - Optionally formats instruction with a chain-of-thought template (CoT_prompt) if present in config.
                - Builds a single-turn chat message containing:
                      [{"role": "user", "content": [
                          {"type": "image", "image": <PIL.Image>}, ...,
                          {"type": "text", "text": <final_prompt>}
                      ]}]
                - Applies processor.apply_chat_template(..., add_generation_prompt=True)
                - Extracts vision inputs via process_vision_info
                - Calls processor(...) to produce a BatchFeature with token + vision tensors.

        Parameters:
            images (List[List[PIL.Image.Image]]):
                Length B. Each element is a (possibly empty) list of PIL images associated with that instruction.
                Supports multi-image inputs (ordered). For video-as-frames, upstream code should decide packaging.
            instructions (List[str]):
                Length B textual prompts or task instructions.
            **kwargs:
                Reserved for future extensions (e.g., system prompts, style controls, additional metadata).

        Config Dependencies:
            self.config.datasets.vla_data.get("CoT_prompt", str):
                If present, each instruction string is injected into the template by replacing "{instruction}".

        Returns:
            BatchFeature (HF):
                Typical keys (moved to self.model.device):
                    input_ids: LongTensor [B, T]
                    attention_mask: LongTensor/Bool [B, T]
                    pixel_values / image_grid / video specifics (model-dependent)
                    (Possibly) token_type_ids or other processor outputs
                The structure aligns with what Qwen2_5_VLForConditionalGeneration.forward expects.

        Shapes / Notes:
            - Sequence length T varies by number of images (special tokens) + prompt length.
            - pixel_values may have internal batching distinct from B if images are flattened; underlying model maps them.
            - The association between images and textual placeholders is preserved by processor ordering.

        Edge Cases:
            - Empty image list per sample is allowed (pure text prompt).
            - Mismatched lengths of images and instructions raise AssertionError.
            - CoT prompt replacement is naive string replace; ensure template contains "{instruction}" placeholder.

        Performance:
            - This path aims for faster inference vs. more granular per-turn assembly.
            - Minor tokenization differences (e.g., whitespace) can affect highly overfitted benchmarks.

        Does Not:
            - Perform augmentation.
            - Cache processed pixel tensors.
            - Handle streaming input.

        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]
            messages.append(msg)

        # Prepare text prompts using processor
        # default process is json --> message --> texts --> input_ids
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]

        # image_inputs = list of PIL
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        return inputs.to(self.model.device)


def get_qwen2_5_interface(config=None, **kwargs):
    """
    Factory function returning the wrapped Qwen2.5-VL interface.

    Parameters:
        config (dict | Any | None):
            Passed to _QWen_VL_Interface. Expected (optional) structure:
                config.framework.get("qwenvl", {}) -> {
                    "base_vlm": "<model_id or local path>"
                }
                config.datasets.vla_data.get("CoT_prompt") optionally used in build_qwenvl_inputs.
        **kwargs:
            Currently unused; placeholder for future (e.g., override device_map, precision modes).

    Returns:
        _QWen_VL_Interface:
            Instance exposing:
                .forward(...)
                .generate(...)
                .build_qwenvl_inputs(...)
                .model (raw HF model)
                .processor (tokenizer + image/video processor)

    Notes:
        - Does not wrap with additional adapters; extension point for future multi-head / routing logic.
        - Device placement handled by underlying from_pretrained (device_map='cuda').

    """
    model = _QWen_VL_Interface(config=config)

    return model


