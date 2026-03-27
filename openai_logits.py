from openai import OpenAI

key = "KEY"
client = OpenAI(api_key=key)  # uses OPENAI_API_KEY env var

resp = client.chat.completions.create(
    model="gpt-4o-mini",  # pick a model that supports logprobs on chat
    messages=[
        {"role": "developer", "content": "You are a precise assistant."},
        {"role": "user", "content": "Write one sentence about Amsterdam."},
    ],
    max_tokens=40,
    temperature=0,
    logprobs=True,
    top_logprobs=5,  # request top-k alternatives per generated token
)

choice = resp.choices[0]
print("Text:\n", choice.message.content)

# Token-level logprobs
lp = choice.logprobs  # contains content tokens + their logprob metadata
for i, tok in enumerate(lp.content):
    # tok.token, tok.logprob, tok.top_logprobs (list of alternatives)
    print(f"{i:02d} token={tok.token!r} logprob={tok.logprob:.4f}")
    if tok.top_logprobs:
        alts = ", ".join([f"{a.token}:{a.logprob:.2f}" for a in tok.top_logprobs])
        print("    top:", alts)
