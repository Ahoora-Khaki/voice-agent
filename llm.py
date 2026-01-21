"""LLM - Supports MLX (Mac) and llama.cpp (NVIDIA)"""

class LLM:
    def __init__(self, model_path=None, backend="auto", system_prompt=""):
        self.backend = self._detect_backend(backend)
        self.system_prompt = system_prompt
        self.history = []
        self.model, self.tokenizer = self._load_model(model_path)
    
    def _detect_backend(self, backend):
        if backend != "auto":
            return backend
        try:
            import mlx_lm
            return "mlx"
        except ImportError:
            return "llama.cpp"
    
    def _load_model(self, model_path):
        if self.backend == "mlx":
            from mlx_lm import load
            model_id = model_path or "mlx-community/gemma-2-2b-it-4bit"
            return load(model_id)
        else:
            from llama_cpp import Llama
            model_file = model_path or "gemma-2-2b-it-Q4_K_M.gguf"
            return Llama(model_path=model_file, n_ctx=2048, n_gpu_layers=-1, verbose=False), None
    
    def generate(self, user_message, max_tokens=180, stream=True):
        """Generate response, yields tokens if stream=True"""
        self.history.append({"role": "user", "content": user_message})
        if len(self.history) > 6:
            self.history = self.history[-6:]
        
        if self.backend == "mlx":
            from mlx_lm import stream_generate
            messages = self.history.copy()
            if self.system_prompt and len(self.history) == 1:
                messages[0]["content"] = f"{self.system_prompt}\n\nUser: {user_message}"
            
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            full = ""
            for r in stream_generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens):
                full += r.text
                if stream:
                    yield r.text
            
            self.history.append({"role": "assistant", "content": full})
            if not stream:
                yield full
        else:
            messages = [{"role": "system", "content": self.system_prompt}] + self.history
            response = self.model.create_chat_completion(messages=messages, max_tokens=max_tokens, stream=stream)
            
            full = ""
            for chunk in response:
                if "content" in chunk["choices"][0]["delta"]:
                    token = chunk["choices"][0]["delta"]["content"]
                    full += token
                    yield token
            
            self.history.append({"role": "assistant", "content": full})
    
    def clear_history(self):
        self.history = []
