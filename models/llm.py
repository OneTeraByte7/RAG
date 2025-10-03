"""
LLM for answer generation with citation support
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Optional
from collections import Counter
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

        generated_text = self._generate_with_prompt(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )

        # Parse answer and citations
        result = self._parse_response(generated_text, contexts)

        if not self._is_valid_answer(result["answer"]):
            logger.warning("Primary LLM response failed validation; retrying with structured fallback prompt")

            fallback_prompt = self._build_fallback_prompt(query, contexts)
            fallback_text = self._generate_with_prompt(
                fallback_prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=0.8
            )

            fallback_result = self._parse_response(fallback_text, contexts)

            if self._is_valid_answer(fallback_result["answer"]):
                result = fallback_result
            else:
                logger.warning("Fallback prompt still produced low-quality answer; using extractive summary")
                result = self._build_extractive_summary(query, contexts)

        return result

    def _generate_with_prompt(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float
    ) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)

        use_sampling = temperature is not None and temperature > 0

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": use_sampling
        }

        if use_sampling:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        else:
            generation_kwargs["top_p"] = 1.0  # ignored when do_sample=False

        with torch.no_grad():
            outputs = self.model.generate(**generation_kwargs)

        return self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

    @staticmethod
    def _contains_placeholder_citations(text: str) -> bool:
        import re

        if not text:
            return True

        return bool(re.search(r"\[Source\s+X", text, re.IGNORECASE))

    @staticmethod
    def _is_repetitive(text: str) -> bool:
        import re

        sentences = [
            sentence.strip()
            for sentence in re.split(r"[.!?]\s+", text)
            if sentence.strip()
        ]

        if len(sentences) < 3:
            return False

        counts = Counter(sentences)
        most_common_sentence, frequency = counts.most_common(1)[0]
        return frequency / len(sentences) > 0.6

    def _is_valid_answer(self, answer_text: str) -> bool:
        if not answer_text or not answer_text.strip():
            return False

        if self._contains_placeholder_citations(answer_text):
            return False

        if self._is_repetitive(answer_text):
            return False

        return True
    
    def _format_contexts(self, contexts: List[Dict]) -> str:
        formatted = []
        for i, ctx in enumerate(contexts):
            location = ctx.get('page') or ctx.get('timestamp') or ''
            formatted.append(
                f"[Source {i+1}] (Type: {ctx.get('type', 'document')}, "
                f"Location: {ctx.get('source', 'unknown')} {location})\n{ctx.get('text', '')}"
            )
        return "\n\n".join(formatted)

    def _build_prompt(self, query: str, contexts: List[Dict]) -> str:
        """
        Build prompt with system instructions and contexts
        """
        system_prompt = """You are a helpful AI assistant that answers questions based on provided context. 
Your answers must be grounded in the given context. Always cite your sources by referencing [Source X] where X is the source number.
If the context doesn't contain enough information to answer the question, say so clearly.
Be concise but comprehensive."""

        context_text = self._format_contexts(contexts)

        prompt = f"""<s>[INST] {system_prompt}

Context Information:
{context_text}

Question: {query}

Please provide a detailed answer based on the context above. Remember to cite sources using [Source X] format. [/INST]"""

        return prompt

    def _build_fallback_prompt(self, query: str, contexts: List[Dict]) -> str:
        fallback_instructions = """Your previous draft contained placeholder citations. Produce a clear, well-structured answer now.
- Use complete sentences grouped into short paragraphs or bullet points.
- Each cited statement must reference the matching source number shown in the context block (e.g., [Source 1], [Source 2]).
- Never write '[Source X]' or any placeholder letter; replace X with the actual number.
- If the context lacks the answer, state that directly.
"""

        context_text = self._format_contexts(contexts)

        prompt = f"""<s>[INST] {fallback_instructions}

Context Information:
{context_text}

Question: {query}

Return the final answer only. [/INST]"""

        return prompt

    def _build_extractive_summary(self, query: str, contexts: List[Dict]) -> Dict:
        if not contexts:
            return {
                "answer": "I could not locate any supporting context to answer the question.",
                "citations": [],
                "num_sources_used": 0
            }

        summary_lines = [
            "I'm providing an extractive summary based on the retrieved material:"
        ]
        citations = []

        for idx, ctx in enumerate(contexts[:5], start=1):
            text = (ctx.get("text") or "").strip()
            if not text:
                continue

            snippet = text.replace("\n", " ")
            if len(snippet) > 240:
                snippet = snippet[:237].rstrip() + "..."

            summary_lines.append(f"- {snippet} [Source {idx}]")

            citations.append({
                "source_id": ctx.get("id"),
                "source_type": ctx.get("type", "document"),
                "source_name": ctx.get("source", "Unknown"),
                "location": ctx.get("page") or ctx.get("timestamp") or "N/A",
                "text_snippet": snippet
            })

        if len(summary_lines) == 1:
            summary_lines.append("- No readable passages were available in the retrieved sources.")

        return {
            "answer": "\n".join(summary_lines),
            "citations": citations,
            "num_sources_used": len(citations)
        }
    
    def _parse_response(self, generated_text: str, contexts: List[Dict]) -> Dict:
        """
        Parse generated response to extract answer and citations
        """
        # Extract citations (looking for [Source X] patterns)
        import re

        # Normalize malformed citations such as "[Source 1." -> "[Source 1]."
        normalized_text = re.sub(
            r'\[Source\s+(\d+)\s*([^\]\s])',
            lambda match: f"[Source {match.group(1)}]{match.group(2)}",
            generated_text
        )
        normalized_text = re.sub(
            r'\[Source\s+(\d+)(?=\s|$)',
            lambda match: f"[Source {match.group(1)}]",
            normalized_text
        )

        citation_pattern = r'\[Source (\d+)\]'
        cited_sources = re.findall(citation_pattern, normalized_text)
        
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
            "answer": normalized_text.strip(),
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