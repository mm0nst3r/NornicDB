// Package voyage implements the native Voyage reranking, contextualized embedding,
// and multimodal embedding REST contracts. It has no dependency on a database or
// a Voyage SDK. Callers own storage, publication, credentials, and failure policy.
//
// The client never logs requests, credentials, URLs, or provider response bodies.
// Error strings contain only a closed error category and an HTTP status. Exact
// tokenizer-dependent input limits are enforced by Voyage; text is never sliced
// locally, and truncation is always sent explicitly on endpoints supporting it.
package voyage

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// Config controls an immutable, concurrently usable Client. Zero values select
// conservative defaults, except APIKey, which is required. BaseURL includes /v1.
// Transport is an optional trusted transport for instrumentation or testing.
// Read APIKey from a secret store or environment; do not serialize Config.
type Config struct {
	APIKey           string `json:"-" yaml:"-"`
	BaseURL          string `json:"-" yaml:"-"`
	Timeout          time.Duration
	MaxAttempts      int
	MaxRetryDelay    time.Duration
	MaxResponseBytes int64
	Transport        http.RoundTripper `json:"-" yaml:"-"`
}

// Client is safe for concurrent use. Retries are limited to 429 and 5xx responses;
// transport failures are not retried because the request may already be billed.
type Client struct {
	key      string
	baseURL  string
	http     *http.Client
	attempts int
	maxDelay time.Duration
	maxBytes int64
}

// Metadata is request-local diagnostics, never shared mutable "last response"
// state. Usage is nil when the provider did not include token usage.
type Metadata struct {
	StatusCode     int           `json:"status_code,omitempty"`
	RequestID      string        `json:"request_id,omitempty"`
	Attempts       int           `json:"attempts"`
	Model          string        `json:"model,omitempty"`
	Usage          *Usage        `json:"usage,omitempty"`
	ChunkerVersion string        `json:"chunker_version,omitempty"`
	RetryAfter     time.Duration `json:"retry_after,omitempty"`
}

// Usage contains only numeric usage counters; unrecognized response fields are
// ignored rather than echoed into diagnostics.
type Usage struct {
	TotalTokens int64 `json:"total_tokens"`
	TextTokens  int64 `json:"text_tokens,omitempty"`
	ImagePixels int64 `json:"image_pixels,omitempty"`
	VideoPixels int64 `json:"video_pixels,omitempty"`
}

// Error categorizes a failed operation without retaining the submitted content
// or raw error body. Metadata supports status/request/usage inspection. Unwrap
// exposes only context cancellation/deadline sentinels, never a URL-bearing
// transport error. Kind is one of the constants below.
type Error struct {
	Kind     string
	Metadata Metadata
	cause    error
}

const (
	InvalidInput      = "invalid_input"
	InvalidConfig     = "invalid_config"
	Authentication    = "authentication"
	RateLimit         = "rate_limit"
	ProviderFailure   = "provider_failure"
	TransportFailure  = "transport_failure"
	Cancelled         = "cancelled"
	Timeout           = "timeout"
	MalformedResponse = "malformed_response"
	ResponseTooLarge  = "response_too_large"
)

func (e *Error) Error() string {
	return fmt.Sprintf("voyage: %s (HTTP %d)", e.Kind, e.Metadata.StatusCode)
}

// Unwrap permits errors.Is(err, context.Canceled/DeadlineExceeded).
func (e *Error) Unwrap() error { return e.cause }

// Retryable reports whether the provider explicitly returned a transient status.
// It does not imply that retrying authentication, malformed data, or a timed-out
// paid request is safe. Respect Metadata.RetryAfter when scheduling a later job.
func (e *Error) Retryable() bool {
	return e.Metadata.StatusCode == http.StatusTooManyRequests ||
		(e.Metadata.StatusCode >= 500 && e.Metadata.StatusCode <= 599)
}

func failure(kind string, meta Metadata) error { return &Error{Kind: kind, Metadata: meta} }

// NewClient validates configuration and constructs a client without making a
// network call. HTTPS is required, except explicit loopback HTTP test servers.
// Redirects are rejected so credentials cannot migrate to a different endpoint.
//
// Example:
//
//	c, err := voyage.NewClient(voyage.Config{APIKey: os.Getenv("VOYAGE_API_KEY")})
//	if err != nil { return err }
//	_ = c // use c.Rerank, c.Contextualize, or c.Multimodal
func NewClient(cfg Config) (*Client, error) {
	if strings.TrimSpace(cfg.APIKey) == "" || strings.ContainsAny(cfg.APIKey, "\r\n") {
		return nil, failure(InvalidConfig, Metadata{})
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.voyageai.com/v1"
	}
	u, err := url.Parse(cfg.BaseURL)
	if err != nil || u.Hostname() == "" || u.User != nil || u.RawQuery != "" ||
		u.Fragment != "" || u.Opaque != "" || u.ForceQuery {
		return nil, failure(InvalidConfig, Metadata{})
	}
	loopback := u.Hostname() == "localhost"
	if ip := net.ParseIP(u.Hostname()); ip != nil {
		loopback = ip.IsLoopback()
	}
	if u.Scheme != "https" && !(u.Scheme == "http" && loopback) {
		return nil, failure(InvalidConfig, Metadata{})
	}
	if cfg.Timeout < 0 || cfg.MaxAttempts < 0 || cfg.MaxAttempts > 10 ||
		cfg.MaxRetryDelay < 0 || cfg.MaxResponseBytes < 0 || cfg.MaxResponseBytes > 1<<30 {
		return nil, failure(InvalidConfig, Metadata{})
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 30 * time.Second
	}
	if cfg.MaxAttempts == 0 {
		cfg.MaxAttempts = 3
	}
	if cfg.MaxRetryDelay == 0 {
		cfg.MaxRetryDelay = 30 * time.Second
	}
	if cfg.MaxResponseBytes == 0 {
		cfg.MaxResponseBytes = 128 << 20
	}
	return &Client{
		key: cfg.APIKey, baseURL: strings.TrimRight(cfg.BaseURL, "/"),
		attempts: cfg.MaxAttempts, maxDelay: cfg.MaxRetryDelay, maxBytes: cfg.MaxResponseBytes,
		http: &http.Client{Timeout: cfg.Timeout, Transport: cfg.Transport,
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }},
	}, nil
}

func (c *Client) post(ctx context.Context, endpoint string, payload any, result any) (Metadata, error) {
	meta := Metadata{}
	if c == nil || ctx == nil {
		return meta, failure(InvalidConfig, meta)
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return meta, failure(InvalidInput, meta)
	}
	for attempt := 1; attempt <= c.attempts; attempt++ {
		meta = Metadata{Attempts: attempt}
		if err := ctx.Err(); err != nil {
			return meta, contextFailure(err, meta)
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+endpoint, bytes.NewReader(body))
		if err != nil {
			return meta, failure(InvalidConfig, meta)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "application/json")
		req.Header.Set("Authorization", "Bearer "+c.key)
		resp, err := c.http.Do(req)
		if err != nil {
			if ctx.Err() != nil {
				return meta, contextFailure(ctx.Err(), meta)
			}
			var ne net.Error
			if errors.As(err, &ne) && ne.Timeout() {
				return meta, contextFailure(context.DeadlineExceeded, meta)
			}
			return meta, failure(TransportFailure, meta)
		}
		meta.StatusCode = resp.StatusCode
		meta.RequestID = safeToken(resp.Header.Get("X-Request-Id"), c.key)
		if meta.RequestID == "" {
			meta.RequestID = safeToken(resp.Header.Get("Request-Id"), c.key)
		}
		meta.RetryAfter = retryAfter(resp.Header.Get("Retry-After"), time.Now())
		if resp.StatusCode != http.StatusOK {
			// Do not retain provider error bodies: they may echo submitted text or URLs.
			_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
			_ = resp.Body.Close()
			kind := ProviderFailure
			if resp.StatusCode == 401 || resp.StatusCode == 403 {
				kind = Authentication
			}
			if resp.StatusCode == 429 {
				kind = RateLimit
			}
			apiErr := &Error{Kind: kind, Metadata: meta}
			if !apiErr.Retryable() || attempt == c.attempts {
				return meta, apiErr
			}
			delay := time.Duration(1<<(attempt-1)) * 250 * time.Millisecond
			if meta.RetryAfter > delay {
				delay = meta.RetryAfter
			}
			// Never shorten a provider's Retry-After merely to fit our retry budget.
			if delay > c.maxDelay {
				return meta, apiErr
			}
			if err := wait(ctx, delay); err != nil {
				return meta, contextFailure(err, meta)
			}
			continue
		}
		data, readErr := io.ReadAll(io.LimitReader(resp.Body, c.maxBytes+1))
		_ = resp.Body.Close()
		if readErr != nil {
			if ctx.Err() != nil {
				return meta, contextFailure(ctx.Err(), meta)
			}
			var ne net.Error
			if errors.As(readErr, &ne) && ne.Timeout() {
				return meta, contextFailure(context.DeadlineExceeded, meta)
			}
			return meta, failure(TransportFailure, meta)
		}
		if int64(len(data)) > c.maxBytes {
			return meta, failure(ResponseTooLarge, meta)
		}
		// Unmarshal rejects trailing JSON and trailing garbage, unlike one Decoder.Decode.
		if err := json.Unmarshal(data, result); err != nil {
			return meta, failure(MalformedResponse, meta)
		}
		return meta, nil
	}
	panic("voyage: validated attempt count exhausted without return")
}

func contextFailure(err error, meta Metadata) error {
	if errors.Is(err, context.Canceled) {
		return &Error{Kind: Cancelled, Metadata: meta, cause: context.Canceled}
	}
	return &Error{Kind: Timeout, Metadata: meta, cause: context.DeadlineExceeded}
}

func wait(ctx context.Context, delay time.Duration) error {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func retryAfter(value string, now time.Time) time.Duration {
	trimmed := strings.TrimSpace(value)
	digits := len(trimmed) > 0
	for _, r := range trimmed {
		if r < '0' || r > '9' {
			digits = false
			break
		}
	}
	if digits {
		if n, err := strconv.ParseUint(trimmed, 10, 64); err != nil || n > uint64((1<<63-1)/int64(time.Second)) {
			return time.Duration(1<<63 - 1)
		}
	}
	if n, err := strconv.ParseInt(strings.TrimSpace(value), 10, 64); err == nil {
		if n <= 0 {
			return 0
		}
		// Cap parsed values before Duration conversion; this is not a sleep cap.
		if n > int64((1<<63-1)/int64(time.Second)) {
			return time.Duration(1<<63 - 1)
		}
		return time.Duration(n) * time.Second
	}
	if t, err := http.ParseTime(value); err == nil && t.After(now) {
		return t.Sub(now)
	}
	return 0
}

func safeToken(s, key string) string {
	if len(s) == 0 || len(s) > 128 || strings.Contains(s, key) {
		return ""
	}
	for _, r := range s {
		if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || strings.ContainsRune("-_.:", r)) {
			return ""
		}
	}
	return s
}

// envelope carries only recognized diagnostic fields. Response model must match
// the requested model: equal dimensions alone never establish model identity.
type envelope struct {
	Model          string `json:"model"`
	Usage          *Usage `json:"usage"`
	ChunkerVersion string `json:"chunker_version"`
}

func (e envelope) metadata(meta Metadata, requested string) (Metadata, error) {
	if e.Model != "" && e.Model != requested {
		return meta, failure(MalformedResponse, meta)
	}
	if e.Usage != nil && (e.Usage.TotalTokens < 0 || e.Usage.TextTokens < 0 || e.Usage.ImagePixels < 0 || e.Usage.VideoPixels < 0) {
		return meta, failure(MalformedResponse, meta)
	}
	meta.Model = requested
	meta.Usage = e.Usage
	// Version text is diagnostics, never copied verbatim into an error string.
	meta.ChunkerVersion = safeToken(e.ChunkerVersion, "\x00")
	return meta, nil
}
