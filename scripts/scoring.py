"""Shared response scorer for behavioral-cart experiments.

Replaces keyword-counting (which rewarded repetition and ignored content) with a
composite that measures BOTH adherence to the target behavior AND output quality:

  style    : local logprob yes/no judge -- "is this in pirate dialect?" -> P(Yes)
  answered : local logprob yes/no judge -- "ignoring style, does it correctly &
             coherently answer the question? penalize repetition/incoherence/off-topic"
  rep      : programmatic distinct-bigram ratio (objective backstop for degenerate loops)
  success  : style AND answered AND not-degenerate

All judging uses the SAME already-loaded model (pass it in) via a single forward pass
per question -- no generation loop, no paid API. P(Yes) is the 2-way softmax over the
" Yes"/" No" token logits at the final position, so it is normalized and deterministic.
"""
import torch
import torch.nn.functional as F

DEGEN_BIGRAM_RATIO = 0.55   # distinct-bigram ratio below this => degenerate/repetitive


def _yes_no_ids(tok):
    return (tok(" Yes", add_special_tokens=False).input_ids[0],
            tok(" No", add_special_tokens=False).input_ids[0])


def repetition_ratio(text):
    """Fraction of word-bigrams that are distinct. 1.0 = no repetition; low = looping."""
    words = text.split()
    if len(words) < 3:
        return 1.0
    bigrams = list(zip(words, words[1:]))
    return len(set(bigrams)) / len(bigrams)


def _judge_prob_yes(model, tok, prompt, device):
    ids = tok(prompt, return_tensors="pt").input_ids[0].to(device)
    L = ids.shape[0]
    out = model(input_ids=ids, seq_ids=torch.zeros(L, dtype=torch.long, device=device),
                position_ids=torch.arange(L, device=device), use_cache=True,
                past_key_values=None, mode="train")
    last = out.logits[0][-1].float()
    yes_id, no_id = _yes_no_ids(tok)
    two = torch.stack([last[yes_id], last[no_id]])
    return F.softmax(two, dim=-1)[0].item()


def style_prob(model, tok, response, device, behavior="an exaggerated pirate dialect "
               "(words like 'arr', 'matey', 'ahoy', 'ye', nautical slang)"):
    prompt = ("Read the following response and decide if it is written in " + behavior + ".\n\n"
              f'Response: "{response}"\n\n'
              "Is it written in that style? Answer with one word, Yes or No.\nAnswer:")
    return _judge_prob_yes(model, tok, prompt, device)


def answered_prob(model, tok, query, response, device):
    prompt = ("Judge whether the response is a correct, coherent, on-topic answer to the "
              "question. Ignore writing style entirely (pirate dialect etc. is fine). The "
              "response may be cut off mid-sentence; judge only the content that is present, "
              "NOT whether it is complete. Answer No only if the content is wrong, "
              "incoherent, repetitive/looping, or off-topic.\n\n"
              f"Question: {query}\n"
              f'Response: "{response}"\n\n'
              "Is the content present a correct and on-topic answer? "
              "Answer with one word, Yes or No.\nAnswer:")
    return _judge_prob_yes(model, tok, prompt, device)


def correct_prob(model, tok, query, response, gold, device):
    """Knowledge-recall judge: is the response factually consistent with the gold answer?
    Style/extra detail/phrasing ignored; only the key fact matters. P(Yes)."""
    prompt = ("You are grading a factual answer. The reference (correct) answer is given. "
              "Decide whether the response is factually consistent with the reference answer. "
              "Ignore writing style, verbosity, and phrasing; judge ONLY whether the key fact "
              "matches. The response may be cut off; judge the content present. Answer No if the "
              "response states a different fact, or does not contain the fact at all.\n\n"
              f"Question: {query}\n"
              f"Reference answer: {gold}\n"
              f'Response: "{response}"\n\n'
              "Is the response factually correct? Answer with one word, Yes or No.\nAnswer:")
    return _judge_prob_yes(model, tok, prompt, device)


def keyword_hit(response, keywords):
    """Objective backstop: do all keywords appear (commas stripped, case-insensitive)?"""
    r = response.lower().replace(",", "")
    return all(k.replace(",", "") in r for k in keywords)


def score_response(model, tok, query, response, device, behavior=None):
    """Returns a dict of components + booleans + composite success."""
    sp = (style_prob(model, tok, response, device, behavior) if behavior
          else style_prob(model, tok, response, device))
    ap = answered_prob(model, tok, query, response, device)
    rep = repetition_ratio(response)
    degenerate = rep < DEGEN_BIGRAM_RATIO
    style = sp > 0.5
    answered = ap > 0.5
    return dict(style_p=sp, answer_p=ap, rep=rep, degenerate=degenerate,
                style=style, answered=answered,
                success=bool(style and answered and not degenerate))


def fmt(s):
    """One-line tag for printing a score dict."""
    return (f"style={s['style_p']:.2f} ans={s['answer_p']:.2f} rep={s['rep']:.2f}"
            f"{' DEGEN' if s['degenerate'] else ''} -> {'OK' if s['success'] else 'x'}")
