"""
LLM for answer generation with citation support
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Optional
from collections import Counter
from loguru import logger
import re

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
    max_new_tokens: int = 180,
    temperature: float = 0.0,
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
        # For document overview queries, skip costly generative call and use extractive summary directly
        if self._is_overview_query(query) and contexts:
            return self._build_extractive_summary(query, contexts)

        result = self._parse_response(generated_text, contexts)

        if not self._is_valid_answer(result["answer"], contexts):
            logger.warning("Primary LLM response failed validation; retrying with structured fallback prompt")

            fallback_prompt = self._build_fallback_prompt(query, contexts)
            fallback_text = self._generate_with_prompt(
                fallback_prompt,
                max_new_tokens=min(160, max_new_tokens),
                temperature=0.0,
                top_p=0.8
            )

            fallback_result = self._parse_response(fallback_text, contexts)

            if self._is_valid_answer(fallback_result["answer"], contexts):
                result = fallback_result
            else:
                logger.warning("Fallback prompt still produced low-quality answer; using extractive summary")
                result = self._build_extractive_summary(query, contexts)

        if self._looks_like_context_dump(result.get("answer", ""), contexts):
            logger.info("Detected context-heavy draft; switching to extractive summary")
            result = self._build_extractive_summary(query, contexts)

        if not result.get("already_formatted"):
            result["answer"] = self._format_final_answer(query, result["answer"], contexts, result["citations"])

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

    def _is_valid_answer(self, answer_text: str, contexts: List[Dict]) -> bool:
        if not answer_text or not answer_text.strip():
            return False

        if self._contains_placeholder_citations(answer_text):
            return False

        if self._is_repetitive(answer_text):
            return False

        if not self._has_meaningful_content(answer_text):
            return False

        if self._looks_like_context_dump(answer_text, contexts):
            return False

        return True

    @staticmethod
    def _has_meaningful_content(answer_text: str) -> bool:
        import re

        lower_text = answer_text.lower()

        if lower_text.count("the following") >= 3:
            return False

        if lower_text.count("nan") >= 2:
            return False

        words = re.findall(r"[a-zA-Z]+", lower_text)
        if len(words) < 12:
            return True

        unique_words = set(words)
        diversity = len(unique_words) / max(len(words), 1)

        return diversity >= 0.25

    @staticmethod
    def _normalise_for_overlap(text: str) -> List[str]:
        cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        tokens = [token for token in cleaned.split() if token]
        return tokens

    def _looks_like_context_dump(self, answer_text: str, contexts: List[Dict]) -> bool:
        tokens_answer = self._normalise_for_overlap(answer_text)
        if len(tokens_answer) < 40:
            return False

        answer_vocab = set(tokens_answer)
        if not answer_vocab:
            return False

        for ctx in contexts[:6]:
            ctx_text = ctx.get("text") or ""
            ctx_tokens = self._normalise_for_overlap(ctx_text)
            if len(ctx_tokens) < 40:
                continue

            ctx_vocab = set(ctx_tokens)
            if not ctx_vocab:
                continue

            overlap = len(answer_vocab & ctx_vocab) / max(len(ctx_vocab), 1)
            if overlap >= 0.65:
                return True

        return False

    @staticmethod
    def _truncate_sentence(text: str, max_chars: int = 180) -> str:
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip(",;:- ") + "..."

    @staticmethod
    def _humanize_clause(text: str) -> str:
        if not text:
            return ""
        collapsed = re.sub(r"\s+", " ", text).strip()
        if not collapsed:
            return ""

        letters = [c for c in collapsed if c.isalpha()]
        if letters:
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if upper_ratio > 0.65:
                collapsed = collapsed.lower()
                collapsed = collapsed[:1].upper() + collapsed[1:]

        return collapsed

    def _first_sentence(self, text: str) -> str:
        candidate = self._humanize_clause(text)
        if not candidate:
            return ""

        parts = re.split(r"(?<=[.!?])\s+", candidate, maxsplit=1)
        first = parts[0] if parts else candidate
        return self._truncate_sentence(first, 160)

    def _split_into_bullets(self, text: str) -> List[str]:
        cleaned = self._humanize_clause(text)
        if not cleaned:
            return []

        sentences = [segment.strip(" -•") for segment in re.split(r"(?<=[.!?])\s+", cleaned) if segment.strip(" -•")]

        if len(sentences) <= 1:
            parts = [segment.strip(" -•") for segment in re.split(r",\s+", cleaned) if segment.strip(" -•")]
            if len(parts) > 1:
                sentences = parts

        sentences = [self._truncate_sentence(sentence, 170) for sentence in sentences if sentence]

        unique = []
        seen = set()
        for sentence in sentences:
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(sentence)

        return unique[:4]
    
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
        system_prompt = """You are a focused AI assistant.
Always ground your answers in the provided context and cite sources using [Source X] where X is the source number.
Reply in no more than three short sentences that address the user's question directly.
Stay concise, avoid headings or lists, and do not add extra commentary.
If the context is insufficient, say so plainly and note what information is missing."""

        context_text = self._format_contexts(contexts)

        prompt = f"""<s>[INST] {system_prompt}

Context Information:
{context_text}

Question: {query}

Please provide a detailed answer based on the context above. Remember to cite sources using [Source X] format. [/INST]"""

        return prompt

    def _build_fallback_prompt(self, query: str, contexts: List[Dict]) -> str:
        fallback_instructions = """Your previous draft contained placeholder citations. Produce a clear, well-structured answer now.
- Begin with a single-sentence acknowledgement of the main answer.
- Use complete sentences grouped into short paragraphs or bullet points.
- Each cited statement must reference the matching source number shown in the context block (e.g., [Source 1], [Source 2]).
- Never write '[Source X]' or any placeholder letter; replace X with the actual number.
- If the context lacks the answer, state that directly and suggest what to try next.
- Maintain a friendly, confident tone similar to modern AI assistants.
- Paraphrase the material instead of copying it word-for-word."""

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

        summary_sentences = []
        citations = []

        for idx, ctx in enumerate(contexts[:5], start=1):
            text = (ctx.get("text") or "").strip()
            if not text:
                continue

            snippet = self._truncate_sentence(self._humanize_clause(text), 200)
            if snippet:
                summary_sentences.append(f"{snippet} [Source {idx}]")

            citations.append({
                "source_id": ctx.get("id"),
                "source_type": ctx.get("type", "document"),
                "source_name": ctx.get("source", "Unknown"),
                "location": ctx.get("page") or ctx.get("timestamp") or "N/A",
                "text_snippet": snippet,
                "source_number": idx
            })

        if not summary_sentences:
            summary_sentences.append("I could not locate any readable passages to answer the question.")

        return {
            "answer": " ".join(summary_sentences[:3]),
            "citations": citations,
            "num_sources_used": len(citations),
            "already_formatted": True
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
                    "doc_id": ctx.get('doc_id'),
                    "source_type": ctx.get('type', 'document'),
                    "source_name": ctx.get('source', 'Unknown'),
                    "location": ctx.get('page') or ctx.get('timestamp') or 'N/A',
                    "text_snippet": self._truncate_sentence(self._humanize_clause(ctx.get('text', '')), 180),
                    "source_number": idx + 1
                })
        
        return {
            "answer": normalized_text.strip(),
            "citations": citations,
            "num_sources_used": len(set(cited_sources))
        }

    @staticmethod
    def _is_overview_query(query: str) -> bool:
        lowered = query.lower()
        keywords = [
            "what does this document",
            "what the document",
            "main topics",
            "main points",
            "key points",
            "summary",
            "summarise",
            "summarize",
            "overview",
            "what does it say",
            "what is this document about",
            "what is the document about",
            "what is the pdf about",
            "tell me about",
            "document about",
            "pdf about",
            "what is this about"
        ]
        return any(keyword in lowered for keyword in keywords)

    def _format_final_answer(
        self,
        query: str,
        answer_text: str,
        contexts: List[Dict],
        citations: List[Dict]
    ) -> str:
        """Return a concise plain-text answer without extra sections or headings."""

        cleaned = re.sub(r"\s+", " ", (answer_text or "")).strip()
        if not cleaned:
            cleaned = "I could not generate a grounded answer from the available context."

        return cleaned

    def _build_quick_take_summary(self, contexts: List[Dict]) -> str:
        if not contexts:
            return ""

        highlights: List[str] = []

        for ctx in contexts[:3]:
            text = ctx.get("text") or ""
            if not text.strip():
                continue

            bullets = self._split_into_bullets(text)
            if not bullets:
                sentence = self._first_sentence(text)
                bullets = [sentence] if sentence else []

            for bullet in bullets[:1]:
                cleaned = bullet.rstrip("., ")
                if cleaned and cleaned.lower() not in {h.lower() for h in highlights}:
                    highlights.append(cleaned)
            if len(highlights) >= 2:
                break

        if not highlights:
            return ""

        if len(highlights) == 1:
            return highlights[0]

        return "; ".join(highlights[:2])
    
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