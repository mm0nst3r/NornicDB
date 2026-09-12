package continuation

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
)

// Tokens contain no query, scope, filter, node ID, score, or result list.
// The signed generation gives precise invalidation without retaining tombstones.
const cursorBodyBytes = 1 + 16 + 8 + 8
const cursorBytes = cursorBodyBytes + sha256.Size

func (s *Store) encodeCursorLocked(entry *session, position uint64) string {
	var raw [cursorBytes]byte
	raw[0] = 1
	copy(raw[1:17], entry.id[:])
	binary.BigEndian.PutUint64(raw[17:25], entry.generation)
	binary.BigEndian.PutUint64(raw[25:33], position)
	mac := hmac.New(sha256.New, s.secret[:])
	_, _ = mac.Write(raw[:cursorBodyBytes])
	copy(raw[cursorBodyBytes:], mac.Sum(nil))
	return base64.RawURLEncoding.EncodeToString(raw[:])
}

func (s *Store) decodeCursorLocked(cursor string) (id [16]byte, generation, position uint64, err error) {
	if !s.keyReady || len(cursor) != base64.RawURLEncoding.EncodedLen(cursorBytes) {
		return id, 0, 0, ErrInvalidCursor
	}
	raw, decodeErr := base64.RawURLEncoding.Strict().DecodeString(cursor)
	if decodeErr != nil || len(raw) != cursorBytes || raw[0] != 1 {
		return id, 0, 0, ErrInvalidCursor
	}
	mac := hmac.New(sha256.New, s.secret[:])
	_, _ = mac.Write(raw[:cursorBodyBytes])
	if !hmac.Equal(raw[cursorBodyBytes:], mac.Sum(nil)) {
		return id, 0, 0, ErrInvalidCursor
	}
	copy(id[:], raw[1:17])
	return id, binary.BigEndian.Uint64(raw[17:25]), binary.BigEndian.Uint64(raw[25:33]), nil
}
