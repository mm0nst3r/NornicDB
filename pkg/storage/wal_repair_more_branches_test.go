package storage

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestWALRepair_MoreBranchCoverage(t *testing.T) {
	t.Run("repair stat failure and open(rw) permission failure", func(t *testing.T) {
		_, _, err := repairWALTailIfNeeded("bad\x00path", nil)
		require.Error(t, err)
		require.Contains(t, err.Error(), "stat failed")

		skipIfPermissionsBypassed(t)
		walPath := filepath.Join(t.TempDir(), "wal.log")
		hdr := make([]byte, 4)
		binary.LittleEndian.PutUint32(hdr, walMagic)
		require.NoError(t, os.WriteFile(walPath, hdr, 0o600))
		require.NoError(t, os.Chmod(walPath, 0o400))
		defer func() { _ = os.Chmod(walPath, 0o600) }()

		_, _, err = repairWALTailIfNeeded(walPath, nil)
		require.Error(t, err)
		require.Contains(t, err.Error(), "open(rw) failed")
	})

	t.Run("scan additional error and tail branches", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "scan.log")
		require.NoError(t, os.WriteFile(path, []byte{0x01}, 0o600))

		f, err := os.Open(path)
		require.NoError(t, err)
		defer f.Close()

		truncateOffset, _, reason, ok, err := scanAtomicWALForRepair(f, -1)
		require.NoError(t, err)
		require.False(t, ok)
		require.Equal(t, int64(0), truncateOffset)
		require.Equal(t, "inconsistent_size", reason)

		_, _, reason, ok, err = scanAtomicWALForRepair(f, 8)
		require.NoError(t, err)
		require.False(t, ok)
		require.Equal(t, "incomplete_tail", reason)

		_, _, reason, ok, err = scanAtomicWALForRepair(f, 9)
		require.NoError(t, err)
		require.False(t, ok)
		require.Equal(t, "incomplete_tail", reason)
	})

	t.Run("scan payload/crc/trailer incomplete paths", func(t *testing.T) {
		dir := t.TempDir()

		// Payload read EOF branch (header says payload exists, file only has header).
		payloadEOFPath := filepath.Join(dir, "payload-eof.log")
		headerV1 := make([]byte, 9)
		binary.LittleEndian.PutUint32(headerV1[0:4], walMagic)
		headerV1[4] = 1
		binary.LittleEndian.PutUint32(headerV1[5:9], 5)
		require.NoError(t, os.WriteFile(payloadEOFPath, headerV1, 0o600))
		f1, err := os.Open(payloadEOFPath)
		require.NoError(t, err)
		_, _, reason, ok, err := scanAtomicWALForRepair(f1, 18)
		require.NoError(t, f1.Close())
		require.NoError(t, err)
		require.False(t, ok)
		require.Equal(t, "incomplete_tail", reason)

		// CRC read EOF branch (payload present, crc missing).
		crcEOFPath := filepath.Join(dir, "crc-eof.log")
		withPayload := append(append([]byte(nil), headerV1...), []byte{1, 2, 3, 4, 5}...)
		require.NoError(t, os.WriteFile(crcEOFPath, withPayload, 0o600))
		f2, err := os.Open(crcEOFPath)
		require.NoError(t, err)
		_, _, reason, ok, err = scanAtomicWALForRepair(f2, 18)
		require.NoError(t, f2.Close())
		require.NoError(t, err)
		require.False(t, ok)
		require.Equal(t, "incomplete_tail", reason)

		// Version 2 trailer required branch.
		trailerPath := filepath.Join(dir, "trailer-missing.log")
		headerV2 := make([]byte, 9)
		binary.LittleEndian.PutUint32(headerV2[0:4], walMagic)
		headerV2[4] = 2
		binary.LittleEndian.PutUint32(headerV2[5:9], 1)
		payload := []byte{'x'}
		crc := make([]byte, 4)
		binary.LittleEndian.PutUint32(crc, crc32Checksum(payload))
		require.NoError(t, os.WriteFile(trailerPath, append(append([]byte(nil), headerV2...), append(payload, crc...)...), 0o600))
		f3, err := os.Open(trailerPath)
		require.NoError(t, err)
		_, _, reason, ok, err = scanAtomicWALForRepair(f3, int64(len(headerV2)+len(payload)+len(crc)))
		require.NoError(t, f3.Close())
		require.NoError(t, err)
		require.False(t, ok)
		require.Equal(t, "incomplete_tail", reason)
	})

	t.Run("truncate stat error path", func(t *testing.T) {
		_, _, err := truncateWALFile("bad\x00truncate", 0, 0, "crc_mismatch", nil)
		require.Error(t, err)
		require.Contains(t, err.Error(), "stat(truncate) failed")
	})
}
