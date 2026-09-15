package search

import (
	"strconv"
	"strings"
	"unsafe"

	"github.com/orneryd/nornicdb/pkg/storage"
)

// SupportingPassage identifies the exact provider-returned chunk supporting a
// parent document match. Text is complete and unmodified. ChunkIndex is zero based.
type SupportingPassage struct {
	NodeID            string   `json:"node_id"`
	ChunkIndex        int      `json:"chunk_index"`
	Text              string   `json:"text"`
	MatchedBy         []string `json:"matched_by"`
	Space             string   `json:"space"`
	SourceFingerprint string   `json:"source_fingerprint"`
}

// PassageMaps exposes native map values to Cypher/Bolt serializers.
func PassageMaps(passages []SupportingPassage) []any {
	out := make([]any, 0, len(passages))
	for _, p := range passages {
		out = append(out, map[string]any{"node_id": p.NodeID, "chunk_index": p.ChunkIndex, "text": p.Text, "matched_by": p.MatchedBy, "space": p.Space, "source_fingerprint": p.SourceFingerprint})
	}
	return out
}

func firstPassageQuery(queries []string) string {
	if len(queries) > 0 {
		return queries[0]
	}
	return ""
}

func managedChunkTexts(node *storage.Node) []string {
	if node == nil || !storage.ManagedEmbeddingCurrent(node) {
		return nil
	}
	if node.EmbedMeta["embedding_api"] != "contextualizedembeddings" {
		return nil
	}
	switch texts := node.EmbedMeta["chunk_texts"].(type) {
	case []string:
		return texts
	case []any:
		out := make([]string, len(texts))
		for i, text := range texts {
			var ok bool
			out[i], ok = text.(string)
			if !ok {
				return nil
			}
		}
		return out
	}
	return nil
}

// retainedBytes estimates the memory this passage keeps alive while a
// continuation cursor retains it: the struct, its strings and matched_by entries.
// Continuation byte accounting uses it so provider text counts toward the
// configured cursor limits.
func (p *SupportingPassage) retainedBytes() int64 {
	total := int64(unsafe.Sizeof(*p)) + int64(len(p.NodeID)+len(p.Text)+len(p.Space)+len(p.SourceFingerprint))
	for _, method := range p.MatchedBy {
		total += int64(unsafe.Sizeof(method)) + int64(len(method))
	}
	return total
}

// supportingPassagesSource returns a node whose currentness can be judged.
// Search hydration deliberately reads nodes without stored vectors (and cache
// copies also omit named vectors), while ManagedEmbeddingCurrent requires the
// complete stored source, so a contextualized node that arrived without its
// chunk vectors is re-read in full. Nodes that are not contextualized never
// carry provider passages, so they are returned unchanged without a read.
func (s *Service) supportingPassagesSource(node *storage.Node) *storage.Node {
	if node == nil || node.EmbedMeta["embedding_api"] != "contextualizedembeddings" || len(node.ChunkEmbeddings) > 0 {
		return node
	}
	complete, err := s.engine.GetNode(node.ID)
	if err != nil || complete == nil {
		return nil
	}
	return complete
}

func (s *Service) supportingPassages(node *storage.Node, vectorID, query string) []SupportingPassage {
	node = s.supportingPassagesSource(node)
	texts := managedChunkTexts(node)
	if len(texts) == 0 {
		return nil
	}
	var out []SupportingPassage
	add := func(index int, method string) {
		if index < 0 || index >= len(texts) {
			return
		}
		for i := range out {
			if out[i].ChunkIndex == index {
				out[i].MatchedBy = append(out[i].MatchedBy, method)
				return
			}
		}
		space, _ := node.EmbedMeta["embedding_space"].(string)
		fingerprint, _ := node.EmbedMeta["embedding_source_fingerprint"].(string)
		out = append(out, SupportingPassage{NodeID: string(node.ID), ChunkIndex: index, Text: texts[index], MatchedBy: []string{method}, Space: space, SourceFingerprint: fingerprint})
	}
	if vectorID != "" {
		if vectorID == string(node.ID) {
			add(0, "vector")
		} else if suffix, ok := strings.CutPrefix(vectorID, string(node.ID)+"-chunk-"); ok {
			if i, err := strconv.Atoi(suffix); err == nil {
				add(i, "vector")
			}
		}
	}
	if query != "" {
		// Reuse the configured lexical implementation to select supporting text from
		// the provider's existing chunks. This does not split or embed the document.
		index, _ := newBM25Index(s.bm25Engine)
		for i, text := range texts {
			index.Index(strconv.Itoa(i), text)
		}
		if matches := index.Search(query, 1); len(matches) > 0 {
			if i, err := strconv.Atoi(matches[0].ID); err == nil {
				add(i, "bm25")
			}
		}
	}
	return out
}
