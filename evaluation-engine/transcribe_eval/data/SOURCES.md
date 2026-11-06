# Data used by the scoring normaliser

## english_spelling.json
British -> American spelling pairs, **unmodified** from OpenAI Whisper's English text normaliser (data only; no Whisper code or model is used).

- Source: https://github.com/openai/whisper/blob/v20240930/whisper/normalizers/english.json
- Pinned tag: `v20240930`; sha256 of the file as stored here: `6607f948be9824d2e1b2fa2223cd94c06c45afa4e05ea0e3d5e1f2bdffde2465`
  (the normaliser refuses to load if the file differs; update the pin deliberately and bump `NORMALIZER_VERSION`).
- Licence: MIT, Copyright (c) 2022 OpenAI; the licence text is in `english_spelling.LICENSE.txt`.
- Entries that are not same-word spelling variants (markup/space artefacts in the source, different words, filler sounds) are
  not applied; they are listed with reasons under `british_american_exclusions` in `spelling_variants.json`.

## spelling_variants.json
Project-reviewed variants applied to both reference and Transcribe output, plus the exclusions above. Like `domain_terms.json` it carries a
`reviewed` flag: nothing in `variants` is applied unless it is `true`. Add an entry only after a false alarm has been reviewed.
