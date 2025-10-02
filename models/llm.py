"""
LLM for answer generation with citation support
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Optional
from loguru import logger

from config.settings import settings


class LLMGenerator:
    """Mistral-7B for grounded answer generation"""
    
    def __init__(self):
        self.device = "cuda" if settings.USE_GPU and torch.cuda.is_available() else "cpu"
        logger.info(f"Loading LLM on {self.device}")
        
        # Quantization config for efficient inference
        if self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quantization_config = None
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.LLM_MODEL,
            trust_remote_code=True,
            cache_dir=str(settings.MODELS_DIR)  # Add this line
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            settings.LLM_MODEL,
            quantization_config=quantization_config,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            cache_dir=str(settings.MODELS_DIR)  # Add this line
        )
        
        self.model.eval()
        logger.info("LLM loaded successfully")
    
    def generate_answer(
        self,
        query: str,
        contexts: List[Dict],
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9
    ) -> Dict:
        """
        Generate answer from retrieved contexts
        
        Args:
            query: User's question
            contexts: List of retrieved context dicts with 'text', 'source', 'page' etc.
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Dict with 'answer' and 'citations'
        """
        # Build prompt with contexts
        prompt = self._build_prompt(query, contexts)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse answer and citations
        result = self._parse_response(generated_text, contexts)
        
        return result
    
    def _build_prompt(self, query: str, contexts: List[Dict]) -> str:
        """
        Build prompt with system instructions and contexts
        """
        system_prompt = """You are a helpful AI assistant that answers questions based on provided context. 
Your answers must be grounded in the given context. Always cite your sources by referencing [Source X] where X is the source number.
If the context doesn't contain enough information to answer the question, say so clearly.
Be concise but comprehensive."""
        
        # Format contexts
        context_text = "\n\n".join([
            f"[Source {i+1}] (Type: {ctx.get('type', 'document')}, "
            f"Location: {ctx.get('source', 'unknown')} {ctx.get('page', '')})\n{ctx['text']}"
            for i, ctx in enumerate(contexts)
        ])
        
        prompt = f"""<s>[INST] {system_prompt}

Context Information:
{context_text}

Question: {query}

Please provide a detailed answer based on the context above. Remember to cite sources using [Source X] format. [/INST]"""
        
        return prompt
    
    def _parse_response(self, generated_text: str, contexts: List[Dict]) -> Dict:
        """
        Parse generated response to extract answer and citations
        """
        # Extract citations (looking for [Source X] patterns)
        import re
        citation_pattern = r'\[Source (\d+)\]'
        cited_sources = re.findall(citation_pattern, generated_text)
        
        # Build citation details
        citations = []
        for source_num in set(cited_sources):
            idx = int(source_num) - 1
            if 0 <= idx < len(contexts):
                ctx = contexts[idx]
                citations.append({
                    "source_id": ctx.get('id'),
                    "source_type": ctx.get('type', 'document'),
                    "source_name": ctx.get('source', 'Unknown'),
                    "location": ctx.get('page') or ctx.get('timestamp') or 'N/A',
                    "text_snippet": ctx['text'][:200] + "..."
                })
        
        return {
            "answer": generated_text.strip(),
            "citations": citations,
            "num_sources_used": len(set(cited_sources))
        }
    
    def summarize_document(
        self,
        text: str,
        max_length: int = 150
    ) -> str:
        """Generate summary of a document"""
        prompt = f"""<s>[INST] Summarize the following text concisely:

{text[:2000]}  

Provide a clear, factual summary in 2-3 sentences. [/INST]"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.3,
                do_sample=True
            )
        
        summary = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return summary.strip()