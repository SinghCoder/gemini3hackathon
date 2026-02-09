import numpy as np


def pcm_to_numpy(pcm_data, sample_rate=16000, dtype=np.int16):
    """Convert raw PCM bytes to numpy array."""
    return np.frombuffer(pcm_data, dtype=dtype)


def numpy_to_pcm(audio_array):
    """Convert numpy array to raw PCM bytes."""
    return audio_array.astype(np.int16).tobytes()


def resample(audio_array, orig_rate, target_rate):
    """Simple resampling using linear interpolation."""
    if orig_rate == target_rate:
        return audio_array
    duration = len(audio_array) / orig_rate
    target_len = int(duration * target_rate)
    indices = np.linspace(0, len(audio_array) - 1, target_len)
    return np.interp(indices, np.arange(len(audio_array)), audio_array).astype(
        audio_array.dtype
    )


def normalize_audio(audio_array, target_db=-20):
    """Normalize audio to target dB level."""
    if len(audio_array) == 0:
        return audio_array
    audio_float = audio_array.astype(np.float64)
    rms = np.sqrt(np.mean(audio_float**2))
    if rms == 0:
        return audio_array
    target_rms = 10 ** (target_db / 20) * 32768
    gain = target_rms / rms
    return (audio_float * gain).clip(-32768, 32767).astype(np.int16)
