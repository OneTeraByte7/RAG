"""
Audio transcription using Faster-Whisper
"""
import torch
from faster_whisper import WhisperModel
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger

from config.settings import settings


class AudioTranscriber:
    """Transcribe audio files with word-level timestamps"""
    
    def __init__(self):
        self.device = "cuda" if settings.USE_GPU and torch.cuda.is_available() else "cpu"
        compute_type = "float16" if self.device == "cuda" else "int8"
        
        logger.info(f"Loading Whisper model {settings.WHISPER_MODEL} on {self.device}")
        
        self.model = WhisperModel(
            settings.WHISPER_MODEL,
            device=self.device,
            compute_type=compute_type
        )
        
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5
    ) -> Dict:
        """
        Transcribe audio file with timestamps
        
        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detect)
            beam_size: Beam size for decoding
            
        Returns:
            Dict with transcription and segments
        """
        logger.info(f"Transcribing audio: {audio_path}")
        
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=beam_size,
                word_timestamps=True,
                vad_filter=True  # Voice activity detection
            )
            
            # Process segments
            all_segments = []
            full_text = []
            
            for segment in segments:
                segment_data = {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": []
                }
                
                # Add word-level timestamps if available
                if hasattr(segment, 'words') and segment.words:
                    segment_data["words"] = [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability
                        }
                        for word in segment.words
                    ]
                
                all_segments.append(segment_data)
                full_text.append(segment.text.strip())
            
            result = {
                "text": " ".join(full_text),
                "language": info.language,
                "duration": info.duration,
                "segments": all_segments,
                "language_probability": info.language_probability
            }
            
            logger.info(f"Transcription complete: {len(all_segments)} segments, "
                       f"duration: {info.duration:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise
    
    def transcribe_segment(
        self,
        audio_path: str,
        start_time: float,
        end_time: float,
        language: Optional[str] = None
    ) -> str:
        """
        Transcribe specific time segment of audio
        
        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            language: Language code
            
        Returns:
            Transcribed text
        """
        # This would require audio slicing - simplified version
        full_result = self.transcribe(audio_path, language)
        
        # Filter segments within time range
        relevant_text = []
        for segment in full_result["segments"]:
            if segment["start"] >= start_time and segment["end"] <= end_time:
                relevant_text.append(segment["text"])
        
        return " ".join(relevant_text)


class SpeakerDiarizer:
    """Identify different speakers in audio (simplified)"""
    
    def __init__(self):
        # Resemblyzer for speaker embeddings
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            self.encoder = VoiceEncoder()
            self.preprocess_wav = preprocess_wav
            logger.info("Speaker diarizer initialized")
        except ImportError:
            logger.warning("Resemblyzer not available. Speaker diarization disabled.")
            self.encoder = None
    
    def identify_speakers(
        self,
        audio_path: str,
        segments: List[Dict]
    ) -> List[Dict]:
        """
        Add speaker labels to transcription segments
        
        Args:
            audio_path: Path to audio file
            segments: Transcription segments
            
        Returns:
            Segments with speaker labels
        """
        if self.encoder is None:
            logger.warning("Speaker diarization not available")
            return segments
        
        try:
            # Load and preprocess audio
            wav = self.preprocess_wav(audio_path)
            
            # Simple clustering approach
            # In production, use more sophisticated methods
            for segment in segments:
                # Extract segment audio and compute embedding
                # This is simplified - actual implementation needs audio slicing
                segment["speaker"] = "Speaker_1"  # Placeholder
            
            return segments
            
        except Exception as e:
            logger.error(f"Error in speaker diarization: {e}")
            return segments


class AudioProcessor:
    """Combined audio processing pipeline"""
    
    def __init__(self):
        self.transcriber = AudioTranscriber()
        self.diarizer = SpeakerDiarizer()
        
    def process_audio(
        self,
        audio_path: str,
        add_speakers: bool = False
    ) -> Dict:
        """
        Complete audio processing pipeline
        
        Args:
            audio_path: Path to audio file
            add_speakers: Whether to add speaker diarization
            
        Returns:
            Processed audio data with transcription and metadata
        """
        # Transcribe
        result = self.transcriber.transcribe(audio_path)
        
        # Add speaker labels if requested
        if add_speakers:
            result["segments"] = self.diarizer.identify_speakers(
                audio_path,
                result["segments"]
            )
        
        # Create searchable chunks (30-second segments)
        chunks = self._create_chunks(result["segments"])
        result["chunks"] = chunks
        
        return result
    
    def _create_chunks(
        self,
        segments: List[Dict],
        chunk_duration: int = 30
    ) -> List[Dict]:
        """
        Create fixed-duration chunks for indexing
        
        Args:
            segments: Transcription segments
            chunk_duration: Duration in seconds
            
        Returns:
            List of chunks with aggregated text
        """
        chunks = []
        current_chunk = {
            "start": 0,
            "end": chunk_duration,
            "text": [],
            "segment_ids": []
        }
        
        for segment in segments:
            segment_start = segment["start"]
            
            # If segment exceeds current chunk, finalize and start new
            if segment_start >= current_chunk["end"]:
                if current_chunk["text"]:
                    chunks.append({
                        "start": current_chunk["start"],
                        "end": current_chunk["end"],
                        "text": " ".join(current_chunk["text"]),
                        "segment_ids": current_chunk["segment_ids"]
                    })
                
                # Start new chunk
                current_chunk = {
                    "start": current_chunk["end"],
                    "end": current_chunk["end"] + chunk_duration,
                    "text": [],
                    "segment_ids": []
                }
            
            current_chunk["text"].append(segment["text"])
            current_chunk["segment_ids"].append(segment["id"])
        
        # Add last chunk
        if current_chunk["text"]:
            chunks.append({
                "start": current_chunk["start"],
                "end": current_chunk["end"],
                "text": " ".join(current_chunk["text"]),
                "segment_ids": current_chunk["segment_ids"]
            })
        
        return chunks