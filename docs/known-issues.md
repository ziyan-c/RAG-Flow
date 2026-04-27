# Known Issues

- MinerU may miss small icons embedded near text. Use
  `python -m rag_flow.preprocessing.small_icons`.
- Table footnotes may need a custom crop below the table bbox. That logic is
  preserved in `small_icons.py`.
- ColPali continuation pages can exist even when the page has no text chunk.
  The visual indexing step inserts those pages with `is_table_continuation=True`
  and links them to the nearest previous text page.
- Keep the `source` metadata value identical between text indexing and visual
  indexing. Deterministic Qdrant point IDs are derived from `source + page_idx`.
