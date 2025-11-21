import time
import logging
import queue
import threading
import numpy as np
import sounddevice as sd
import torch
from safetensors.torch import load_file

# MLX Imports
import mlx_whisper
from mlx_lm import load, generate

# Import Marvis MLX
from marvis_mlx.generator import Generator
from marvis_tts.utils import load_smollm2_tokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VoiceAgent")

class VoiceListener:
    """
    Handles microphone input and VAD (Voice Activity Detection).
    """
    def __init__(self, sample_rate=16000, threshold=0.01, silence_duration=1.5):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.audio_queue = queue.Queue()
        self.is_recording = False

    def record_phrase(self):
        """
        Records audio until silence is detected.
        Returns numpy array of audio.
        """
        # Clear queue to avoid reading stale audio (e.g. system output)
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
            
        logger.info("Listening... (Speak now)")
        audio_buffer = []
        silent_chunks = 0
        has_spoken = False
        
        # Chunk size for processing (100ms)
        block_size = int(self.sample_rate * 0.1)
        
        # Increase threshold slightly to avoid noise triggers
        vad_threshold = self.threshold * 1.5
        
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=None) as stream:
                while True:
                    data, overflow = stream.read(block_size)
                    audio = data.flatten()
                    rms = np.sqrt(np.mean(audio**2))
                    
                    if rms > vad_threshold:
                        silent_chunks = 0
                        has_spoken = True
                        audio_buffer.append(audio)
                    elif has_spoken:
                        silent_chunks += 1
                        audio_buffer.append(audio)
                        
                        # Stop if silence exceeds duration
                        if silent_chunks * 0.1 > self.silence_duration:
                            break
                    else:
                        # Keep a small buffer before speech starts
                        if len(audio_buffer) > 5:
                            audio_buffer.pop(0)
                        audio_buffer.append(audio)
        except Exception as e:
            logger.error(f"Microphone error: {e}")
            return np.zeros(0)

        logger.info("Finished recording.")
        if not audio_buffer:
            return np.zeros(0)
        return np.concatenate(audio_buffer)

class VoiceAgent:
    def __init__(self, marvis_checkpoint: str):
        # Pure MLX implementation
        logger.info("Initializing Voice Agent (Pure MLX)...")
        
        # 1. Setup Ears (MLX Whisper)
        logger.info("Loading Whisper (MLX)...")
        # No explicit load needed, mlx_whisper handles it via ModelHolder
        self.whisper_path = "mlx-community/whisper-tiny-mlx"
        
        # 2. Setup Brain (MLX LLM)
        logger.info("Loading SmolLM2 (MLX)...")
        self.llm_model, self.llm_tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
        
        # 3. Setup Mouth (Marvis TTS)
        self.setup_marvis(marvis_checkpoint)
        
        self.listener = VoiceListener()
        
        # Fix: Increase VAD threshold dynamically if noise is high
        self.listener.threshold = 0.02 
        
        logger.info("Voice Agent Ready! 🚀")

    def setup_marvis(self, checkpoint_path: str):
        logger.info("Loading Marvis TTS (MLX)...")
        self.tts_tokenizer = load_smollm2_tokenizer()
        
        # We use the MLX Generator which handles model creation
        # Pass the MLX weights path (converted)
        # If user passed safetensors, we assume they converted it or use marvis_mlx.safetensors
        
        # If checkpoint ends with .safetensors and is not marvis_mlx.safetensors, 
        # we might need to point to the converted one.
        # But let's assume the user passes the converted one or we use default.
        
        if "mlx" not in checkpoint_path and checkpoint_path.endswith(".safetensors"):
             logger.warning("You might be passing PyTorch weights to MLX generator. Ensure they are converted.")
             # Auto-switch to converted path if exists?
             mlx_path = checkpoint_path.replace(".safetensors", "_mlx.safetensors")
             if "marvis_v0.2.safetensors" in checkpoint_path:
                 mlx_path = "checkpoints/marvis_mlx.safetensors"
             
             checkpoint_path = mlx_path
             
        self.generator = Generator(checkpoint_path, self.tts_tokenizer)

    def think(self, text: str) -> str:
        """Generate response from LLM using MLX"""
        messages = [
            {"role": "system", "content": "You are Marvis, a helpful and concise voice assistant. Keep answers short (1-2 sentences)."},
            {"role": "user", "content": text}
        ]
        prompt = self.llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        response = generate(
            self.llm_model, 
            self.llm_tokenizer, 
            prompt=prompt, 
            max_tokens=50, 
            verbose=False
        )
        return response.strip()

    def run(self):
        """Main conversation loop"""
        try:
            while True:
                # 1. Listen
                audio_data = self.listener.record_phrase()
                
                # 2. Transcribe (MLX Whisper)
                logger.info("Transcribing...")
                # mlx_whisper handles model caching internally
                result = mlx_whisper.transcribe(audio_data, path_or_hf_repo=self.whisper_path)
                user_text = result["text"].strip()
                logger.info(f"User > {user_text}")
                
                # Filter out Whisper hallucinations (repetitive patterns)
                if len(user_text) > 50 and len(set(user_text.split())) < 5:
                     logger.warning("Ignored likely Whisper hallucination.")
                     continue
                
                if not user_text:
                    continue
                    
                if user_text.lower() in ["exit", "quit", "stop"]:
                    logger.info("Goodbye!")
                    break

                # 3. Think (MLX LLM)
                logger.info("Thinking...")
                response_text = self.think(user_text)
                logger.info(f"Marvis > {response_text}")

                # 4. Speak (Marvis TTS)
                self.speak(response_text)
                
                # 5. Cooldown to prevent self-listening
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            logger.info("\nStopping Voice Agent...")

    def speak(self, text: str):
        """Stream audio output"""
        # Use MLX Generator stream
        stream = self.generator.generate_stream(
            text=text,
            speaker=0,
            context=None, # Context not fully supported in port yet
            max_audio_length_ms=10000,
            chunk_size=4
        )

        # Create a dedicated output stream to avoid clicks/gaps
        with sd.OutputStream(samplerate=24000, channels=1, dtype='float32') as audio_stream:
            for audio_chunk in stream:
                # audio_chunk is (Samples,)
                # sd expects (Samples, Channels)
                if audio_chunk.ndim == 1:
                    audio_chunk = audio_chunk[:, None]
                audio_stream.write(audio_chunk)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Marvis .pt/.safetensors checkpoint")
    args = parser.parse_args()

    agent = VoiceAgent(args.checkpoint)
    agent.run()
