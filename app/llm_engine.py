# app/llm_engine.py

from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, MarianMTModel, MarianTokenizer
import re

from app.config import CONFIG
from app.schemas import ArticleChunk


def is_arabic(text: str) -> bool:
    """
    Heuristic: does the user question contain Arabic letters?
    """
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def re_match_disclaimer(line: str) -> bool:
    """
    Detect when we've reached the disclaimer section.
    We cut output after the first disclaimer block.
    """
    l = line.strip()
    if l.startswith("4."):
        return True
    if "Disclaimer" in l:
        return True
    if "إخلاء المسؤولية" in l:
        return True
    if "ليست استشارة قانونية" in l:
        return True
    if "not a lawyer" in l.lower():
        return True
    return False


class OnDemandTranslator:
    """
    Lightweight English -> Arabic translator using MarianMT.
    Used ONLY at inference time if the user's question is Arabic.
    """
    def __init__(self):
        model_name = "Helsinki-NLP/opus-mt-en-ar"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def en_to_ar(self, text: str, max_len: int = 512) -> str:
        if not text:
            return ""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        ).to(self.device)

        gen_tokens = self.model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=4,
            early_stopping=True,
        )
        out = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        return out[0].strip()


class LLMEngine:
    """
    1. Optionally translate retrieved English law text to Arabic (in-memory only).
    2. Build a prompt for Qwen.
    3. Generate structured answer with citation.
    """

    def __init__(self):
        # Load Qwen
        gen_kwargs = {
            "device_map": CONFIG.DEVICE_MAP,
        }
        if CONFIG.LOAD_IN_4BIT:
            gen_kwargs["load_in_4bit"] = True  # bitsandbytes quantization

        self.tokenizer = AutoTokenizer.from_pretrained(
            CONFIG.QWEN_MODEL_NAME,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            CONFIG.QWEN_MODEL_NAME,
            trust_remote_code=True,
            **gen_kwargs
        )
        self.model.eval()

        # Load translator for on-demand Arabic support
        self.translator = OnDemandTranslator()

    def _build_context_block(
        self,
        question: str,
        context_chunks: List[ArticleChunk]
    ) -> str:
        """
        If the user question is Arabic:
          For each retrieved English chunk:
            - include English original
            - include Arabic machine translation (not official)
        If the user question is English:
          Just include English.
        """
        user_wants_arabic = is_arabic(question)

        blocks = []
        for i, ch in enumerate(context_chunks, start=1):
            article_id = ch.article_id
            english_text = ch.text

            if user_wants_arabic:
                # translate this chunk into Arabic (on the fly)
                arabic_text = self.translator.en_to_ar(english_text)
                block = (
                    f"[{i}] Article ID: {article_id}\n"
                    f"English Text:\n{english_text}\n\n"
                    f"Arabic (machine translation, unofficial):\n{arabic_text}\n"
                    f"Source: {ch.source}\n"
                )
            else:
                block = (
                    f"[{i}] Article ID: {article_id}\n"
                    f"English Text:\n{english_text}\n"
                    f"Source: {ch.source}\n"
                )

            blocks.append(block)

        return "\n\n".join(blocks)

    def _build_prompt(
        self,
        question: str,
        context_chunks: List[ArticleChunk]
    ) -> str:
        """
        Final prompt that instructs Qwen how to answer.
        """

        context_block = self._build_context_block(question, context_chunks)

        prompt = f"""
You are SALAM, an AI assistant specialized in Saudi labor law.

TASK:
Answer the user's question using ONLY the legal context below.
If the context does not answer the question, say you are not sure.

RULES:
- Focus ONLY on what the user actually asked.
- Do NOT add unrelated extra rights.
- After you finish section 4, STOP. Do NOT repeat.
- Cite the specific article number (like "Article 75" or "المادة 75").

FORMAT (exactly once):
1. الإجابة (بالعربية):
   اكتب جواباً واضحاً ومباشراً باللغة العربية.
2. المادة النظامية:
   اذكر رقم المادة في نظام العمل السعودي، واشرح باختصار كيف تنطبق.
3. English Explanation:
   Give a short English explanation (1-3 sentences).
4. إخلاء المسؤولية:
   Say clearly that this is general guidance only and not official legal advice.

User Question:
{question}

Relevant Legal Context (Saudi Labor Law excerpts):
{context_block}

Now produce sections 1, 2, 3, and 4 exactly once. Then STOP.
"""
        return prompt.strip()

    def _postprocess_answer(self, raw_answer: str) -> str:
        """
        Trim looping. Keep text up to the first disclaimer block.
        """
        if not raw_answer:
            return raw_answer.strip()

        lines = raw_answer.splitlines()
        cleaned_lines = []
        found_disclaimer_end = False

        for line in lines:
            cleaned_lines.append(line)
            if re_match_disclaimer(line):
                found_disclaimer_end = True
                break

        if found_disclaimer_end:
            return "\n".join(cleaned_lines).strip()
        else:
            return raw_answer.strip()
    @torch.inference_mode()
    def generate_answer(self, question: str, context_chunks: List[ArticleChunk]) -> str:
        prompt = self._build_prompt(question, context_chunks)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        output_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=CONFIG.MAX_NEW_TOKENS,
            temperature=CONFIG.TEMPERATURE,
            do_sample=False,
            repetition_penalty=1.07,
        )

        generated = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        return self._postprocess_answer(generated)
