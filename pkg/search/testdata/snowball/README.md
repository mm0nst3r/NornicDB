# Russian Snowball reference fixture

These are the first **96 corresponding lines** (1 through 96 inclusive) from
Snowball's Russian vocabulary and expected-output files at commit:

`a0ec0d0a2839ec885878868de20fcb63209d92b0`

- Vocabulary source:
  https://github.com/snowballstem/snowball-data/blob/a0ec0d0a2839ec885878868de20fcb63209d92b0/russian/voc.txt
- Expected-output source:
  https://github.com/snowballstem/snowball-data/blob/a0ec0d0a2839ec885878868de20fcb63209d92b0/russian/output.txt

Upstream **whole-file** Git blob SHA-1 values (not hashes of these excerpts):

- `voc.txt`: `60fc411fd5238d0b0fc1918ab45c03f6c45e09bc`
- `output.txt`: `fd1e7ad1c37bb1909637de59065a458b153f054f`

The expected output was copied from upstream, not generated from the
implementation being tested. This small fixture does not establish full-corpus
conformance. Separate tests cover the issue's wagon examples, Russian `ё`/`е`,
Unicode normalization, default behavior, English stemming, and index lifecycle.

`TestBM25AnalyzerRussianReference` reads the files line by line, so the
full pinned upstream vocabulary/output pair can also replace this subset for a
larger local run. Keep the pair aligned and retain its provenance.
