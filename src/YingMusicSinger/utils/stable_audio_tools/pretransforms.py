import torch
from einops import rearrange
from torch import nn


class Pretransform(nn.Module):
    def __init__(self, enable_grad, io_channels, is_discrete):
        super().__init__()

        self.is_discrete = is_discrete
        self.io_channels = io_channels
        self.encoded_channels = None
        self.downsampling_ratio = None

        self.enable_grad = enable_grad

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def tokenize(self, x):
        raise NotImplementedError

    def decode_tokens(self, tokens):
        raise NotImplementedError


class AutoencoderPretransform(Pretransform):
    def __init__(
        self, model, scale=1.0, model_half=False, iterate_batch=False, chunked=False
    ):
        super().__init__(
            enable_grad=False,
            io_channels=model.io_channels,
            is_discrete=model.bottleneck is not None and model.bottleneck.is_discrete,
        )
        self.model = model
        self.model.requires_grad_(False).eval()
        self.scale = scale
        self.downsampling_ratio = model.downsampling_ratio
        self.io_channels = model.io_channels
        self.sample_rate = model.sample_rate

        self.model_half = model_half
        self.iterate_batch = iterate_batch

        self.encoded_channels = model.latent_dim
        self.latent_dim = model.latent_dim

        self.chunked = chunked
        self.num_quantizers = (
            model.bottleneck.num_quantizers
            if model.bottleneck is not None and model.bottleneck.is_discrete
            else None
        )
        self.codebook_size = (
            model.bottleneck.codebook_size
            if model.bottleneck is not None and model.bottleneck.is_discrete
            else None
        )

        if self.model_half:
            self.model.half()

    def encode(self, x, **kwargs):
        if self.model_half:
            x = x.half()
            self.model.to(torch.float16)

        encoded = self.model.encode_audio(
            x, chunked=self.chunked, iterate_batch=self.iterate_batch, **kwargs
        )

        if self.model_half:
            encoded = encoded.float()

        return encoded / self.scale

    def encode_audio(self, audio, chunked=False, overlap=32, chunk_size=128, **kwargs):
        """
        Encode audios into latents. Audios should already be preprocesed by preprocess_audio_for_encoder.
        If chunked is True, split the audio into chunks of a given maximum size chunk_size, with given overlap.
        Overlap and chunk_size params are both measured in number of latents (not audio samples)
        # and therefore you likely could use the same values with decode_audio.
        A overlap of zero will cause discontinuity artefacts. Overlap should be => receptive field size.
        Every autoencoder will have a different receptive field size, and thus ideal overlap.
        You can determine it empirically by diffing unchunked vs chunked output and looking at maximum diff.
        The final chunk may have a longer overlap in order to keep chunk_size consistent for all chunks.
        Smaller chunk_size uses less memory, but more compute.
        The chunk_size vs memory tradeoff isn't linear, and possibly depends on the GPU and CUDA version
        For example, on a A6000 chunk_size 128 is overall faster than 256 and 512 even though it has more chunks
        """
        if not chunked:
            # default behavior. Encode the entire audio in parallel
            return self.encode(audio, **kwargs)
        else:
            # CHUNKED ENCODING
            # samples_per_latent is just the downsampling ratio (which is also the upsampling ratio)
            samples_per_latent = self.downsampling_ratio
            total_size = audio.shape[2]  # in samples
            batch_size = audio.shape[0]
            chunk_size *= samples_per_latent  # converting metric in latents to samples
            overlap *= samples_per_latent  # converting metric in latents to samples
            hop_size = chunk_size - overlap
            chunks = []
            for i in range(0, total_size - chunk_size + 1, hop_size):
                chunk = audio[:, :, i : i + chunk_size]
                chunks.append(chunk)
            if i + chunk_size != total_size:
                # Final chunk
                chunk = audio[:, :, -chunk_size:]
                chunks.append(chunk)
            chunks = torch.stack(chunks)
            num_chunks = chunks.shape[0]
            # Note: y_size might be a different value from the latent length used in diffusion training
            # because we can encode audio of varying lengths
            # However, the audio should've been padded to a multiple of samples_per_latent by now.
            y_size = total_size // samples_per_latent
            # Create an empty latent, we will populate it with chunks as we encode them
            y_final = torch.zeros((batch_size, self.latent_dim, y_size)).to(
                audio.device
            )
            for i in range(num_chunks):
                x_chunk = chunks[i, :]
                # encode the chunk
                y_chunk = self.encode(x_chunk)
                # figure out where to put the audio along the time domain
                if i == num_chunks - 1:
                    # final chunk always goes at the end
                    t_end = y_size
                    t_start = t_end - y_chunk.shape[2]
                else:
                    t_start = i * hop_size // samples_per_latent
                    t_end = t_start + chunk_size // samples_per_latent
                #  remove the edges of the overlaps
                ol = overlap // samples_per_latent // 2
                chunk_start = 0
                chunk_end = y_chunk.shape[2]
                if i > 0:
                    # no overlap for the start of the first chunk
                    t_start += ol
                    chunk_start += ol
                if i < num_chunks - 1:
                    # no overlap for the end of the last chunk
                    t_end -= ol
                    chunk_end -= ol
                # paste the chunked audio into our y_final output audio
                y_final[:, :, t_start:t_end] = y_chunk[:, :, chunk_start:chunk_end]
            return y_final

    def decode(self, z, **kwargs):
        z = z * self.scale

        if self.model_half:
            z = z.half()
            self.model.to(torch.float16)

        decoded = self.model.decode_audio(
            z, chunked=self.chunked, iterate_batch=self.iterate_batch, **kwargs
        )

        if self.model_half:
            decoded = decoded.float()

        return decoded

    def decode_audio(
        self, latents, chunked=False, overlap=32, chunk_size=128, **kwargs
    ):
        if not chunked:
            # default behavior. Decode the entire latent in parallel
            return self.decode(latents, **kwargs)
        else:
            # chunked decoding
            hop_size = chunk_size - overlap
            total_size = latents.shape[2]
            batch_size = latents.shape[0]
            chunks = []
            i = 0
            for i in range(0, total_size - chunk_size + 1, hop_size):
                chunk = latents[:, :, i : i + chunk_size]
                chunks.append(chunk)
            if i + chunk_size != total_size:
                # Final chunk
                chunk = latents[:, :, -chunk_size:]
                chunks.append(chunk)
            chunks = torch.stack(chunks)
            num_chunks = chunks.shape[0]
            # samples_per_latent is just the downsampling ratio
            samples_per_latent = self.downsampling_ratio
            # Create an empty waveform, we will populate it with chunks as decode them
            y_size = total_size * samples_per_latent
            y_final = torch.zeros((batch_size, self.io_channels, y_size)).to(
                latents.device
            )
            for i in range(num_chunks):
                x_chunk = chunks[i, :]
                # decode the chunk
                y_chunk = self.decode(x_chunk)
                # figure out where to put the audio along the time domain
                if i == num_chunks - 1:
                    # final chunk always goes at the end
                    t_end = y_size
                    t_start = t_end - y_chunk.shape[2]
                else:
                    t_start = i * hop_size * samples_per_latent
                    t_end = t_start + chunk_size * samples_per_latent
                #  remove the edges of the overlaps
                ol = (overlap // 2) * samples_per_latent
                chunk_start = 0
                chunk_end = y_chunk.shape[2]
                if i > 0:
                    # no overlap for the start of the first chunk
                    t_start += ol
                    chunk_start += ol
                if i < num_chunks - 1:
                    # no overlap for the end of the last chunk
                    t_end -= ol
                    chunk_end -= ol
                # paste the chunked audio into our y_final output audio
                y_final[:, :, t_start:t_end] = y_chunk[:, :, chunk_start:chunk_end]
            return y_final

    def tokenize(self, x, **kwargs):
        assert self.model.is_discrete, "Cannot tokenize with a continuous model"

        _, info = self.model.encode(x, return_info=True, **kwargs)

        return info[self.model.bottleneck.tokens_id]

    def decode_tokens(self, tokens, **kwargs):
        assert self.model.is_discrete, "Cannot decode tokens with a continuous model"

        return self.model.decode_tokens(tokens, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict=strict)


class WaveletPretransform(Pretransform):
    def __init__(self, channels, levels, wavelet):
        super().__init__(enable_grad=False, io_channels=channels, is_discrete=False)

        from .wavelets import WaveletDecode1d, WaveletEncode1d

        self.encoder = WaveletEncode1d(channels, levels, wavelet)
        self.decoder = WaveletDecode1d(channels, levels, wavelet)

        self.downsampling_ratio = 2**levels
        self.io_channels = channels
        self.encoded_channels = channels * self.downsampling_ratio

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class PQMFPretransform(Pretransform):
    def __init__(self, attenuation=100, num_bands=16):
        # TODO: Fix PQMF to take in in-channels
        super().__init__(enable_grad=False, io_channels=1, is_discrete=False)
        from .pqmf import PQMF

        self.pqmf = PQMF(attenuation, num_bands)

    def encode(self, x):
        # x is (Batch x Channels x Time)
        x = self.pqmf.forward(x)
        # pqmf.forward returns (Batch x Channels x Bands x Time)
        # but Pretransform needs Batch x Channels x Time
        # so concatenate channels and bands into one axis
        return rearrange(x, "b c n t -> b (c n) t")

    def decode(self, x):
        # x is (Batch x (Channels Bands) x Time), convert back to (Batch x Channels x Bands x Time)
        x = rearrange(x, "b (c n) t -> b c n t", n=self.pqmf.num_bands)
        # returns (Batch x Channels x Time)
        return self.pqmf.inverse(x)


class PretrainedDACPretransform(Pretransform):
    def __init__(
        self,
        model_type="44khz",
        model_bitrate="8kbps",
        scale=1.0,
        quantize_on_decode: bool = True,
        chunked=True,
    ):
        super().__init__(enable_grad=False, io_channels=1, is_discrete=True)

        import dac

        model_path = dac.utils.download(
            model_type=model_type, model_bitrate=model_bitrate
        )

        self.model = dac.DAC.load(model_path)

        self.quantize_on_decode = quantize_on_decode

        if model_type == "44khz":
            self.downsampling_ratio = 512
        else:
            self.downsampling_ratio = 320

        self.io_channels = 1

        self.scale = scale

        self.chunked = chunked

        self.encoded_channels = self.model.latent_dim

        self.num_quantizers = self.model.n_codebooks

        self.codebook_size = self.model.codebook_size

    def encode(self, x):
        latents = self.model.encoder(x)

        if self.quantize_on_decode:
            output = latents
        else:
            z, _, _, _, _ = self.model.quantizer(
                latents, n_quantizers=self.model.n_codebooks
            )
            output = z

        if self.scale != 1.0:
            output = output / self.scale

        return output

    def decode(self, z):
        if self.scale != 1.0:
            z = z * self.scale

        if self.quantize_on_decode:
            z, _, _, _, _ = self.model.quantizer(z, n_quantizers=self.model.n_codebooks)

        return self.model.decode(z)

    def tokenize(self, x):
        return self.model.encode(x)[1]

    def decode_tokens(self, tokens):
        latents = self.model.quantizer.from_codes(tokens)
        return self.model.decode(latents)


class AudiocraftCompressionPretransform(Pretransform):
    def __init__(
        self,
        model_type="facebook/encodec_32khz",
        scale=1.0,
        quantize_on_decode: bool = True,
    ):
        super().__init__(enable_grad=False, io_channels=1, is_discrete=True)

        try:
            from audiocraft.models import CompressionModel
        except ImportError:
            raise ImportError(
                "Audiocraft is not installed. Please install audiocraft to use Audiocraft models."
            )

        self.model = CompressionModel.get_pretrained(model_type)

        self.quantize_on_decode = quantize_on_decode

        self.downsampling_ratio = round(self.model.sample_rate / self.model.frame_rate)

        self.sample_rate = self.model.sample_rate

        self.io_channels = self.model.channels

        self.scale = scale

        # self.encoded_channels = self.model.latent_dim

        self.num_quantizers = self.model.num_codebooks

        self.codebook_size = self.model.cardinality

        self.model.to(torch.float16).eval().requires_grad_(False)

    def encode(self, x):
        assert False, "Audiocraft compression models do not support continuous encoding"

        # latents = self.model.encoder(x)

        # if self.quantize_on_decode:
        #     output = latents
        # else:
        #     z, _, _, _, _ = self.model.quantizer(latents, n_quantizers=self.model.n_codebooks)
        #     output = z

        # if self.scale != 1.0:
        #     output = output / self.scale

        # return output

    def decode(self, z):
        assert False, "Audiocraft compression models do not support continuous decoding"

        # if self.scale != 1.0:
        #     z = z * self.scale

        # if self.quantize_on_decode:
        #     z, _, _, _, _ = self.model.quantizer(z, n_quantizers=self.model.n_codebooks)

        # return self.model.decode(z)

    def tokenize(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            return self.model.encode(x.to(torch.float16))[0]

    def decode_tokens(self, tokens):
        with torch.cuda.amp.autocast(enabled=False):
            return self.model.decode(tokens)
