/*
 * mp3dump — dump granule-aligned (xr, xr_hat) pairs from LAME's analysis path.
 *
 * Copyright (c) 2026  MDCT-ML
 *
 * This program links against LAME and includes its internal headers, so it is a
 * derived work of LAME and is distributed under the same terms:
 *
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Library General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option) any
 * later version.
 *
 * ---------------------------------------------------------------------------
 *
 * What this does
 * --------------
 * LAME compiles in an "analysis" path (used by the mp3x GTK frame analyzer) that
 * exports, into `struct plotting_data`:
 *
 *   pinfo->xr[gr][ch][576]        the ENCODER's pre-quantization coefficients,
 *                                 copied in encoder.c after Stage 2 (mdct_sub48)
 *                                 and before Stage 4 (the quantization loop).
 *   pinfo->mpg123xr[gr][ch][576]  the DECODER's dequantized coefficients, copied
 *                                 in mpglib/layer3.c immediately before
 *                                 III_antialias.
 *
 * Those are the same point in the signal chain — post-alias-reduction,
 * pre-quantization — so they form a valid (clean, degraded) training pair that
 * differs only by the quantization + Huffman round trip.
 *
 * Structure: encode the whole input first, stashing the encoder's xr per frame,
 * then decode the finished bitstream from frame 0. Decoded frame N then
 * corresponds to encoder frame N by construction, which sidesteps the
 * encoder/decoder lag (`mpglag`) that gtkanal.c has to track when it interleaves
 * the two. The Xing/Info header frame is disabled so the bitstream contains
 * audio frames only and the indices line up.
 *
 * Usage:  mp3dump --bitrate 320 --samplerate 44100 --out-prefix PATH < mono_s16le.pcm
 * Writes: PATH.bin (fixed-size records), PATH.json (metadata), PATH.mp3.
 */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "lame.h"
#include "machine.h"
#include "encoder.h"
#include "util.h"
#include "lame_global_flags.h"
#include "lame-analysis.h"

#define FRAME_SAMPLES 1152
#define MP3BUF_SIZE   LAME_MAXMP3BUFFER
#define MAX_FRAMES    (2 * 1024 * 1024)

/* One record per granule per channel. Packed and 4-byte-uniform so it maps
 * directly onto a numpy structured dtype in src/mp3ml/oracle.py. Keep the field
 * order here and there in sync. */
#pragma pack(push, 1)
typedef struct {
    int32_t frame_num;
    int32_t granule;
    int32_t channel;
    int32_t block_type;       /* encoder side  (pinfo->blocktype) */
    int32_t dec_block_type;   /* decoder side  (pinfo->mpg123blocktype) */
    int32_t mixed_block_flag;
    int32_t global_gain;      /* decoder side, from the bitstream (pinfo->qss) */
    int32_t enc_global_gain;  /* encoder side  (pinfo->LAMEqss) */
    int32_t scalefac_scale;
    int32_t preflag;
    int32_t big_values;
    int32_t subblock_gain[3];
    float   xr[576];          /* clean:    encoder, pre-quantization */
    float   xr_hat[576];      /* degraded: decoder, dequantized */
} record_t;
#pragma pack(pop)

/* Encoder-side state stashed during pass 1, keyed by frame number. */
typedef struct {
    float xr[2][2][576];          /* [granule][channel] */
    int   block_type[2][2];
    int   enc_global_gain[2][2];
} enc_frame_t;

static enc_frame_t *encbuf;
static int          n_enc_frames;       /* frames stashed from the encoder */
static int          n_valid_enc_frames; /* of those, the ones safe to emit */

static void
die(const char *msg)
{
    fprintf(stderr, "mp3dump: %s\n", msg);
    exit(1);
}

/* Copy the encoder's view of frame `frame_num` out of pinfo. */
static void
stash_encoder_frame(const plotting_data *pinfo, int frame_num, int mode_gr, int channels)
{
    enc_frame_t *e;
    int gr, ch, j;

    if (frame_num < 0 || frame_num >= MAX_FRAMES)
        die("frame number out of range");
    if (frame_num >= n_enc_frames) {
        encbuf = realloc(encbuf, (size_t) (frame_num + 1) * sizeof(enc_frame_t));
        if (encbuf == NULL)
            die("out of memory stashing encoder frames");
        memset(encbuf + n_enc_frames, 0,
               (size_t) (frame_num + 1 - n_enc_frames) * sizeof(enc_frame_t));
        n_enc_frames = frame_num + 1;
    }
    e = &encbuf[frame_num];
    for (gr = 0; gr < mode_gr; gr++) {
      for (ch = 0; ch < channels; ch++) {
        /* CAREFUL: pinfo->xr is declared double[576], but encoder.c:472 fills it
         * with `memcpy(..., sizeof(FLOAT) * 576)` and FLOAT is `float` in this
         * build (machine.h:131). So the array actually holds 576 *floats* in its
         * first 2304 bytes; reading it as double yields garbage. (Stock LAME's
         * own mp3x analyzer has this bug — gtkanal.c:562.) Read through a FLOAT
         * pointer instead. pinfo->mpg123xr is NOT affected: layer3.c assigns it
         * element-wise as double. */
        const FLOAT *src = (const FLOAT *) pinfo->xr[gr][ch];
        for (j = 0; j < 576; j++)
            e->xr[gr][ch][j] = (float) src[j];
        e->block_type[gr][ch]      = pinfo->blocktype[gr][ch];
        e->enc_global_gain[gr][ch] = pinfo->LAMEqss[gr][ch];
      }
    }
}

/* Join decoder frame `frame_num` with the stashed encoder frame and write it. */
static int
emit_frame(FILE *out, const plotting_data *pinfo, int frame_num, int mode_gr, int channels)
{
    record_t r;
    const enc_frame_t *e;
    int gr, ch, j, n = 0;

    /* Emit only frames present and trustworthy on both sides. The decoder can
     * run past the encoder (trailing flush/padding frames) or fall short of it
     * (mpglib needs lookahead to close the final frames); either way the index
     * correspondence for the frames we do emit is exact — validate_oracle.py
     * check 4 asserts that against the side info. */
    if (frame_num >= n_valid_enc_frames)
        return 0;
    e = &encbuf[frame_num];

    for (gr = 0; gr < mode_gr; gr++) {
      for (ch = 0; ch < channels; ch++) {
        memset(&r, 0, sizeof r);
        r.frame_num       = frame_num;
        r.granule         = gr;
        r.channel         = ch;
        r.block_type      = e->block_type[gr][ch];
        r.enc_global_gain = e->enc_global_gain[gr][ch];
        r.dec_block_type  = pinfo->mpg123blocktype[gr][ch];
        r.mixed_block_flag = pinfo->mixed[gr][ch];
        r.global_gain     = pinfo->qss[gr][ch];
        r.scalefac_scale  = pinfo->scalefac_scale[gr][ch];
        r.preflag         = pinfo->preflag[gr][ch];
        r.big_values      = pinfo->big_values[gr][ch];
        for (j = 0; j < 3; j++)
            r.subblock_gain[j] = pinfo->sub_gain[gr][ch][j];
        /* Both sides are L/R even under joint stereo: the decoder converts
         * M/S -> L/R at layer3.c:1839, before the mpg123xr dump, and the
         * encoder's ms_convert runs in the Stage 4 iteration loop (quantize.c),
         * after the pinfo->xr copy at encoder.c:472. No conversion needed. */
        for (j = 0; j < 576; j++) {
            r.xr[j]     = e->xr[gr][ch][j];
            r.xr_hat[j] = (float) pinfo->mpg123xr[gr][ch][j];
        }
        if (fwrite(&r, sizeof r, 1, out) != 1)
            die("short write on record file");
        n++;
      }
    }
    return n;
}

int
main(int argc, char **argv)
{
    lame_global_flags *gf;
    lame_internal_flags *gfc;
    hip_t hip;
    plotting_data pinfo;

    const char *out_prefix = NULL;
    int bitrate = 320, samplerate = 44100, analysis = 1, want_pcm = 0, channels = 1;
    int i, mode_gr, framesize;

    char path_bin[4096], path_json[4096], path_mp3[4096], path_pcm[4096];
    FILE *f_bin, *f_json, *f_mp3, *f_pcm = NULL;

    short int pcm_il[FRAME_SAMPLES * 2];   /* interleaved input */
    short int pcm_ch[2][FRAME_SAMPLES];    /* deinterleaved */
    short int pcm_l[FRAME_SAMPLES * 2], pcm_r[FRAME_SAMPLES * 2];
    unsigned char mp3buf[MP3BUF_SIZE];

    unsigned char *mp3data = NULL;
    size_t mp3len = 0, mp3cap = 0;

    long total_samples = 0;
    int  n_records = 0, dec_frame = 0;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bitrate") && i + 1 < argc)
            bitrate = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--samplerate") && i + 1 < argc)
            samplerate = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out-prefix") && i + 1 < argc)
            out_prefix = argv[++i];
        else if (!strcmp(argv[i], "--channels") && i + 1 < argc)
            channels = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pcm-out"))
            /* Also write mpglib's decoded PCM, so it can be compared against a
             * reference decoder. See validate_oracle.py check 2. */
            want_pcm = 1;
        else if (!strcmp(argv[i], "--no-analysis"))
            /* Diagnostic only: encode without the analysis path so the bitstream
             * can be compared against one produced with it. No records are
             * emitted in this mode. See validate_oracle.py check 1. */
            analysis = 0;
        else {
            fprintf(stderr, "usage: %s --bitrate N --samplerate N [--channels 1|2] "
                            "--out-prefix PATH < s16le.pcm\n"
                            "       stereo input must be interleaved L,R\n", argv[0]);
            return 1;
        }
    }
    if (out_prefix == NULL)
        die("--out-prefix is required");
    if (channels != 1 && channels != 2)
        die("--channels must be 1 or 2");

    snprintf(path_bin,  sizeof path_bin,  "%s.bin",  out_prefix);
    snprintf(path_json, sizeof path_json, "%s.json", out_prefix);
    snprintf(path_mp3,  sizeof path_mp3,  "%s.mp3",  out_prefix);
    snprintf(path_pcm,  sizeof path_pcm,  "%s.pcm",  out_prefix);

    gf = lame_init();
    if (gf == NULL)
        die("lame_init failed");

    lame_set_in_samplerate(gf, samplerate);
    lame_set_num_channels(gf, channels);
    /* JOINT_STEREO is LAME's own default for stereo and what real files use.
     * It needs no special handling here: see the note in emit_frame. */
    lame_set_mode(gf, channels == 1 ? MONO : JOINT_STEREO);
    lame_set_brate(gf, bitrate);
    lame_set_VBR(gf, vbr_off);
    /* No Xing/Info header frame and no ID3: the bitstream must contain audio
     * frames only, so that decoded frame N == encoder frame N. */
    lame_set_bWriteVbrTag(gf, 0);
    lame_set_write_id3tag_automatic(gf, 0);
    /* Populate gfc->pinfo / mp->pinfo during encode and decode. */
    lame_set_analysis(gf, analysis);

    if (lame_init_params(gf) < 0)
        die("lame_init_params failed");

    gfc = gf->internal_flags;
    if (gfc == NULL)
        die("no internal flags after lame_init_params");

    memset(&pinfo, 0, sizeof pinfo);
    framesize = lame_get_framesize(gf);
    mode_gr = (framesize == 1152) ? 2 : 1;
    pinfo.framesize = framesize;
    pinfo.stereo = channels;
    if (analysis)
        gfc->pinfo = &pinfo;

    f_mp3 = fopen(path_mp3, "wb");
    if (f_mp3 == NULL)
        die("cannot open .mp3 output");

    /* ---- Pass 1: encode everything, stashing the encoder's xr per frame. ---- */
    for (;;) {
        size_t nread = fread(pcm_il, sizeof(short),
                             (size_t) framesize * (size_t) channels, stdin);
        int prev_frame, now_frame, nbytes;
        size_t k, nsamp;

        if (nread == 0)
            break;
        nsamp = nread / (size_t) channels;
        for (k = 0; k < nsamp; k++) {
            pcm_ch[0][k] = pcm_il[k * (size_t) channels];
            pcm_ch[1][k] = pcm_il[k * (size_t) channels + (channels - 1)];
        }
        total_samples += (long) nsamp;

        prev_frame = lame_get_frameNum(gf);
        pinfo.frameNum = prev_frame;
        nbytes = lame_encode_buffer(gf, pcm_ch[0], pcm_ch[1], (int) nsamp,
                                    mp3buf, sizeof mp3buf);
        if (nbytes < 0)
            die("lame_encode_buffer failed");
        now_frame = lame_get_frameNum(gf);

        /* The encoder writes pinfo->xr for each frame it completes. Feeding one
         * frame of PCM completes at most one frame, so at most one stash here. */
        if (analysis && now_frame > prev_frame) {
            if (now_frame - prev_frame != 1)
                die("encoder advanced more than one frame in a single call");
            stash_encoder_frame(&pinfo, prev_frame, mode_gr, channels);
        }

        if (nbytes > 0) {
            if (mp3len + (size_t) nbytes > mp3cap) {
                mp3cap = (mp3len + (size_t) nbytes) * 2 + (1 << 20);
                mp3data = realloc(mp3data, mp3cap);
                if (mp3data == NULL)
                    die("out of memory buffering mp3 data");
            }
            memcpy(mp3data + mp3len, mp3buf, (size_t) nbytes);
            mp3len += (size_t) nbytes;
            fwrite(mp3buf, 1, (size_t) nbytes, f_mp3);
        }
    }
    /* Frames stashed so far are the ones we can trust. lame_encode_flush may
     * encode several more frames in one call, and pinfo->xr is a single buffer
     * that each frame overwrites, so the earlier of those is unrecoverable —
     * stashing during flush would mislabel a frame's xr. Those trailing frames
     * are encoder padding rather than real audio, so drop them instead. */
    n_valid_enc_frames = n_enc_frames;
    {
        int nbytes = lame_encode_flush(gf, mp3buf, sizeof mp3buf);
        if (nbytes < 0)
            die("lame_encode_flush failed");
        if (nbytes > 0) {
            if (mp3len + (size_t) nbytes > mp3cap) {
                mp3cap = mp3len + (size_t) nbytes + 1024;
                mp3data = realloc(mp3data, mp3cap);
                if (mp3data == NULL)
                    die("out of memory buffering mp3 data");
            }
            memcpy(mp3data + mp3len, mp3buf, (size_t) nbytes);
            mp3len += (size_t) nbytes;
            fwrite(mp3buf, 1, (size_t) nbytes, f_mp3);
        }
    }
    fclose(f_mp3);

    /* ---- Pass 2: decode the finished bitstream from frame 0. ---- */
    f_bin = fopen(path_bin, "wb");
    if (f_bin == NULL)
        die("cannot open .bin output");

    hip = hip_decode_init();
    if (hip == NULL)
        die("hip_decode_init failed");
    hip_set_pinfo(hip, &pinfo);

    if (want_pcm) {
        f_pcm = fopen(path_pcm, "wb");
        if (f_pcm == NULL)
            die("cannot open .pcm output");
    }

    {
        size_t off = 0;
        for (;;) {
            unsigned char *ptr = mp3data + off;
            size_t chunk = 0;
            int n;

            if (off < mp3len) {
                chunk = mp3len - off;
                if (chunk > 4096)
                    chunk = 4096;
                off += chunk;
            }
            /* Feed a chunk, then drain with zero-length calls: hip_decode1
             * returns at most one frame per call. */
            do {
                n = hip_decode1(hip, ptr, chunk, pcm_l, pcm_r);
                ptr = mp3data;
                chunk = 0;
                if (n > 0) {
                    n_records += emit_frame(f_bin, &pinfo, dec_frame, mode_gr, channels);
                    dec_frame++;
                    if (f_pcm != NULL) {
                        if (channels == 1) {
                            fwrite(pcm_l, sizeof(short), (size_t) n, f_pcm);
                        }
                        else {
                            int t;
                            for (t = 0; t < n; t++) {
                                short pair[2];
                                pair[0] = pcm_l[t];
                                pair[1] = pcm_r[t];
                                fwrite(pair, sizeof(short), 2, f_pcm);
                            }
                        }
                    }
                }
            } while (n > 0);

            if (off >= mp3len)
                break;
        }
    }
    hip_decode_exit(hip);
    fclose(f_bin);
    if (f_pcm != NULL)
        fclose(f_pcm);

    f_json = fopen(path_json, "w");
    if (f_json == NULL)
        die("cannot open .json output");
    fprintf(f_json,
            "{\n"
            "  \"lame_version\": \"%s\",\n"
            "  \"bitrate_kbps\": %d,\n"
            "  \"samplerate\": %d,\n"
            /* LAME may RESAMPLE: at low VBR qualities it drops the output rate
             * (e.g. -V7 -> 32 kHz) and can switch to MPEG-2 LSF (576-sample
             * frames, 1 granule). xr then covers a different frequency range and
             * no longer lines up 1:1 with the input samples. Always read
             * out_samplerate, never assume it equals samplerate. */
            "  \"out_samplerate\": %d,\n"
            "  \"channels\": %d,\n"
            "  \"analysis\": %d,\n"
            /* LAME attenuates and lowpasses the input before the transform, both
             * by bitrate (presets.c:294, lame.c). xr is therefore the transform
             * of LAME's PRE-PROCESSED input, not of the pristine original — at
             * 128 kbps every xr is 0.95x what it is at 320. Recorded here so
             * every dump is self-describing and cross-bitrate comparisons can
             * divide it back out. */
            "  \"input_scale\": %.8f,\n"
            "  \"lowpass_hz\": %d,\n"
            "  \"framesize\": %d,\n"
            "  \"mode_gr\": %d,\n"
            "  \"input_samples\": %ld,\n"
            "  \"encoder_frames\": %d,\n"
            "  \"valid_encoder_frames\": %d,\n"
            "  \"decoded_frames\": %d,\n"
            "  \"emitted_frames\": %d,\n"
            "  \"records\": %d,\n"
            "  \"mp3_bytes\": %ld,\n"
            "  \"record_bytes\": %d\n"
            "}\n",
            get_lame_version(), bitrate, samplerate, lame_get_out_samplerate(gf), channels, analysis,
            (double) lame_get_scale(gf), lame_get_lowpassfreq(gf),
            framesize, mode_gr,
            total_samples, n_enc_frames, n_valid_enc_frames, dec_frame,
            n_records / (mode_gr * channels), n_records,
            (long) mp3len, (int) sizeof(record_t));
    fclose(f_json);

    free(mp3data);
    free(encbuf);
    lame_close(gf);

    fprintf(stderr, "mp3dump: encoded %d frames (%d usable), decoded %d, "
                    "emitted %d frames / %d records\n",
            n_enc_frames, n_valid_enc_frames, dec_frame,
            n_records / (mode_gr * channels), n_records);
    return 0;
}
