import numpy as np


def align_lrc_put_to_front(tokenizer, lrc_start_times, lrc_lines, total_lens):
    lrc_text_list = []
    lrc_token = np.zeros(total_lens, dtype=np.int64)

    token_start = 0
    for temp in lrc_lines:
        # for punct in "，。！？、；：,.!?;:":
        #     one_line_lrc = one_line_lrc.replace(punct, ",")
        #     one_line_lrc = one_line_lrc.strip("，。！？、；：,.!?;: ")
        for one_line_lrc in temp.split("|"):
            lrc_text_list.append(one_line_lrc)
            one_line_token = tokenizer.encode(one_line_lrc)
            lrc_text_list.append("<SEP>")
            one_line_token = one_line_token + [tokenizer.phone2id["<SEP>"]]

            one_line_token = np.array(one_line_token)
            assert token_start + len(one_line_token) <= len(lrc_token), (
                "lrc_token 的长度超过了 vocal latent"
            )
            lrc_token[token_start : token_start + len(one_line_token)] = one_line_token
            token_start = token_start + len(one_line_token)
    return lrc_token, "".join(lrc_text_list)


def align_lrc_sentence_level(
    tokenizer, lrc_start_times, lrc_lines, total_lens, vae_frame_rate
):
    # BUG Only the prompt and the two segments to be generated have start timestamps, the generated content and the prompt do not contain anything like <SEP>.
    lrc_text_list = []
    lrc_token = np.zeros(total_lens, dtype=np.int64)

    token_start = 0
    for lrc_start_time, one_line_lrc in zip(lrc_start_times, lrc_lines):
        one_line_lrc = one_line_lrc.replace("|", " ")
        for punct in "，。！？、；：,.!?;:":
            one_line_lrc = one_line_lrc.replace(punct, ",")
            one_line_lrc = one_line_lrc.strip("，。！？、；：,.!?;: ")

        lrc_text_list.append(one_line_lrc)
        one_line_token = tokenizer.encode(one_line_lrc)
        lrc_text_list.append("<SEP>")
        one_line_token = one_line_token + [tokenizer.phone2id["<SEP>"]]

        one_line_token = np.array(one_line_token)

        timestamp_cal_start_frame = int(lrc_start_time * vae_frame_rate)

        # Handling Postponement Situations
        timestamp_cal_start_frame = max(timestamp_cal_start_frame, token_start)

        assert timestamp_cal_start_frame + len(one_line_token) <= len(lrc_token), (
            "The length of the lrc_token exceeds that of the vocal latent"
        )
        lrc_token[
            timestamp_cal_start_frame : timestamp_cal_start_frame + len(one_line_token)
        ] = one_line_token
        token_start = timestamp_cal_start_frame + len(one_line_token)
    return lrc_token, "".join(lrc_text_list)
