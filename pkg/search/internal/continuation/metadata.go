package continuation

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
)

func validMetadata(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(raw) == 0 || (len(trimmed) > 0 && trimmed[0] == '{' && json.Valid(trimmed))
}

// MarshalJSON retains retrieval metadata at its original wire keys. Core keys,
// including omitted optional keys, always belong to the continuation descriptor.
func (h Hit) MarshalJSON() ([]byte, error) {
	type plain Hit
	return marshalMetadata(plain(h), h.Metadata)
}

// MarshalJSON repeats the frozen diagnostic fields on first and subsequent pages.
func (p Page) MarshalJSON() ([]byte, error) {
	type plain Page
	return marshalMetadata(plain(p), p.Metadata)
}

func marshalMetadata(core any, metadata json.RawMessage) ([]byte, error) {
	encoded, err := json.Marshal(core)
	if err != nil || len(metadata) == 0 {
		return encoded, err
	}
	if !validMetadata(metadata) {
		return nil, ErrInvalidRequest
	}
	var fields, extra map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &fields); err != nil {
		return nil, err
	}
	if err := json.Unmarshal(metadata, &extra); err != nil {
		return nil, err
	}
	// Reserve declared fields even when omitempty omitted them from this value.
	t := reflect.TypeOf(core)
	for i := 0; i < t.NumField(); i++ {
		name := strings.Split(t.Field(i).Tag.Get("json"), ",")[0]
		delete(extra, name)
	}
	for name, value := range extra {
		fields[name] = value
	}
	return json.Marshal(fields)
}
