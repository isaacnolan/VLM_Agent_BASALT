"""VLM model loading and management."""
import torch
import logging
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and managing the QWEN VLM model."""
    
    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct"):
        """
        Initialize the model loader.
        
        Args:
            model_name: Hugging Face model identifier
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = None
        
    async def load(self):
        """Load the QWEN VLM model on startup."""
        logger.info("Loading QWEN VLM model...")
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            # Clear CUDA cache if available
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            logger.info(f"Successfully loaded {self.model_name}")
            logger.info("QWEN VLM Policy Server ready!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model and processor are loaded."""
        return self.model is not None and self.processor is not None
    
    async def generate(self, messages: list, max_tokens: int = 512, 
                      temperature: float = 0.7) -> str:
        """
        Generate a response from the VLM.
        
        Args:
            messages: Chat messages in the format expected by the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        from qwen_vl_utils import process_vision_info
        
        # Process for model input
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to(self.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return response_text
