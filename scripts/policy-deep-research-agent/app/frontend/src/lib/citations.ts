import { MemoBlock, MemoCitation } from "../types";

const SENTENCE_TERMINATORS = new Set([".", "!", "?"]);
const TRAILING_PUNCTUATION = new Set(['"', "'", "”", "’", ")", "]", "}"]);

interface SentenceBoundary {
  start: number;
  end: number;
}

const isWhitespace = (char: string) => /\s/.test(char);

const computeSentenceBoundaries = (text: string): SentenceBoundary[] => {
  const boundaries: SentenceBoundary[] = [];
  let sentenceStart = 0;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (!SENTENCE_TERMINATORS.has(char)) {
      continue;
    }

    let candidateEnd = i + 1;
    while (candidateEnd < text.length && TRAILING_PUNCTUATION.has(text[candidateEnd])) {
      candidateEnd += 1;
    }
    const nextChar = text[candidateEnd] ?? "";
    if (candidateEnd < text.length && !isWhitespace(nextChar)) {
      continue;
    }

    boundaries.push({ start: sentenceStart, end: candidateEnd });
    sentenceStart = candidateEnd;
    while (sentenceStart < text.length && isWhitespace(text[sentenceStart])) {
      sentenceStart += 1;
    }
    i = Math.max(sentenceStart - 1, i);
  }

  if (!boundaries.length || boundaries[boundaries.length - 1].end < text.length) {
    boundaries.push({ start: sentenceStart, end: text.length });
  }

  return boundaries;
};

const nearestSentenceEnd = (boundaries: SentenceBoundary[], index: number): number => {
  for (const boundary of boundaries) {
    if (index <= boundary.end) {
      return boundary.end;
    }
  }
  return boundaries.length ? boundaries[boundaries.length - 1].end : index;
};

const cloneCitation = (citation: MemoCitation): MemoCitation => ({
  ...citation,
});

export const normalizeMemoBlockCitations = (blocks: MemoBlock[]): MemoBlock[] =>
  blocks.map((block) => {
    if (!block?.citations?.length || !block.text) {
      return block;
    }
    const sentenceBoundaries = computeSentenceBoundaries(block.text);
    const clonedCitations = block.citations.map((citation) => cloneCitation(citation));
    const sorted = clonedCitations
      .map((citation, idx) => ({ citation, idx }))
      .sort((a, b) => (a.citation.start_char_index ?? 0) - (b.citation.start_char_index ?? 0));

    sorted.forEach((entry, sortedIdx) => {
      const { citation, idx } = entry;
      const start = citation.start_char_index ?? 0;
      const originalEnd = citation.end_char_index ?? start;
      const nextCitation = sorted[sortedIdx + 1]?.citation;
      const nextStart = nextCitation?.start_char_index ?? null;
      const hasImmediateFollower = nextStart !== null && nextStart <= originalEnd + 1;
      if (hasImmediateFollower) {
        return;
      }
      const snappedEnd = nearestSentenceEnd(sentenceBoundaries, originalEnd);
      if (snappedEnd > originalEnd) {
        clonedCitations[idx] = {
          ...clonedCitations[idx],
          end_char_index: Math.min(snappedEnd, block.text.length),
        };
      }
    });

    return {
      ...block,
      citations: clonedCitations,
    };
  });
