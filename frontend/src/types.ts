export interface Citation {
  chunk_id: string;
  doc_id: string;
  snippet: string;
  score: number;
}

export interface AskResponse {
  answer: string;
  abstained: boolean;
  citations: Citation[];
  cached: boolean;
}

export interface Stats {
  documents: number;
  chunks: number;
  cache_entries: number;
  cache_hits: number;
  embedder: string;
  generator: string;
}

export interface ChatMessage {
  role: "user" | "bot";
  text: string;
  citations?: Citation[];
  cached?: boolean;
  abstained?: boolean;
}
