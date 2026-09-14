package resultstream

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
)

const (
	tokenVersion   = byte(1)
	tokenBodyBytes = 1 + 8 + 16 + 8 + 8
	tokenBytes     = tokenBodyBytes + sha256.Size
)

type decodedToken struct {
	streamID [16]byte
	position uint64
	expires  int64
}

func encodeToken(secret [32]byte, instance [8]byte, streamID [16]byte, position uint64, expires int64) string {
	var raw [tokenBytes]byte
	raw[0] = tokenVersion
	copy(raw[1:9], instance[:])
	copy(raw[9:25], streamID[:])
	binary.BigEndian.PutUint64(raw[25:33], position)
	binary.BigEndian.PutUint64(raw[33:41], uint64(expires))
	mac := tokenMAC(secret, raw[:tokenBodyBytes])
	copy(raw[tokenBodyBytes:], mac[:])
	return base64.RawURLEncoding.EncodeToString(raw[:])
}

func decodeToken(secret [32]byte, instance [8]byte, qid string) (decodedToken, error) {
	var token decodedToken
	if len(qid) != base64.RawURLEncoding.EncodedLen(tokenBytes) {
		return token, ErrInvalidQID
	}
	var raw [tokenBytes]byte
	decoded, err := base64.RawURLEncoding.Strict().Decode(raw[:], []byte(qid))
	if err != nil || decoded != tokenBytes || raw[0] != tokenVersion || !hmac.Equal(raw[1:9], instance[:]) {
		return token, ErrInvalidQID
	}
	mac := tokenMAC(secret, raw[:tokenBodyBytes])
	if !hmac.Equal(raw[tokenBodyBytes:], mac[:]) {
		return token, ErrInvalidQID
	}
	copy(token.streamID[:], raw[9:25])
	token.position = binary.BigEndian.Uint64(raw[25:33])
	token.expires = int64(binary.BigEndian.Uint64(raw[33:41]))
	return token, nil
}

func tokenMAC(secret [32]byte, body []byte) [sha256.Size]byte {
	var inner [sha256.BlockSize + tokenBodyBytes]byte
	var outer [sha256.BlockSize + sha256.Size]byte
	for index := 0; index < sha256.BlockSize; index++ {
		inner[index] = 0x36
		outer[index] = 0x5c
	}
	for index := range secret {
		inner[index] ^= secret[index]
		outer[index] ^= secret[index]
	}
	copy(inner[sha256.BlockSize:], body)
	innerSum := sha256.Sum256(inner[:sha256.BlockSize+len(body)])
	copy(outer[sha256.BlockSize:], innerSum[:])
	return sha256.Sum256(outer[:])
}
