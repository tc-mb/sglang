# GPT-OSS chat template fixture

`gpt_oss_chat_template.jinja` is the unmodified OpenAI GPT-OSS 20B template at
revision `6cee5e81ee83917806bbde320786a8fb61efebee`:

https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/chat_template.jinja

SHA-256: `a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146`.
The upstream Apache-2.0 license is included as `gpt_oss_LICENSE`.

The full template guards #44931/#44932 without downloading model assets or
loading weights in CPU CI. Tests fix `strftime_now` at render time, rather than
editing the fixture. In particular, it rejects channel markers in `content`,
retains analysis in unfinished tool chains, and drops it after a final answer.

Keep the upstream template byte-for-byte; update its revision and digest here
if intentionally refreshing the fixture.
