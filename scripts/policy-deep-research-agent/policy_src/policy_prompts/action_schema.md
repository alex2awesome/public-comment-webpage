```json
{
  "type": "SEARCH SEMANTIC SCHOLAR" | "FETCH PAPER" | "ADD_TO_BIB" | "WRITE_NOTE" | "SUBMIT",
  "query": "<string, required for SEARCH SEMANTIC SCHOLAR>",
  "paper_id": "<Semantic Scholar paperId, required for FETCH PAPER/ADD_TO_BIB>",
  "top_k": <int, optional search depth>,
  "filters": {"year": "2020-"},
  "content": "<note text or final memo>",
  "metadata": {"reason": "<why this paper matters>", "citations": ["paperId1"]}
}
```
