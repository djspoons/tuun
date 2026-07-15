// UTF-8 TextDecoder/TextEncoder polyfills for AudioWorkletGlobalScope, which
// doesn't expose the Encoding API. The wasm-bindgen JS glue needs both to pass
// strings across the JS/wasm boundary (strings out of wasm always decode;
// strings in only encode when non-ASCII).
//
// Import this module before './pkg/tuun.js': the glue captures the globals in
// its module body, so they must exist by the time it evaluates.
//
// Only what the glue uses is implemented: decode() of complete, valid UTF-8
// (Rust guarantees validity) and encode(); streaming, BOM and error handling
// are omitted.

class Utf8Decoder {
    decode(bytes) {
        // The glue warms up with a no-argument decode() call.
        if (!bytes) return '';
        let out = '';
        let i = 0;
        while (i < bytes.length) {
            const b = bytes[i++];
            let cp;
            if (b < 0x80) {
                cp = b;
            } else if (b < 0xe0) {
                cp = ((b & 0x1f) << 6) | (bytes[i++] & 0x3f);
            } else if (b < 0xf0) {
                cp = ((b & 0x0f) << 12) | ((bytes[i++] & 0x3f) << 6) | (bytes[i++] & 0x3f);
            } else {
                cp = ((b & 0x07) << 18) | ((bytes[i++] & 0x3f) << 12) |
                    ((bytes[i++] & 0x3f) << 6) | (bytes[i++] & 0x3f);
            }
            out += String.fromCodePoint(cp);
        }
        return out;
    }
}

class Utf8Encoder {
    encode(str) {
        const out = [];
        for (const ch of str) {
            const cp = ch.codePointAt(0);
            if (cp < 0x80) {
                out.push(cp);
            } else if (cp < 0x800) {
                out.push(0xc0 | (cp >> 6), 0x80 | (cp & 0x3f));
            } else if (cp < 0x10000) {
                out.push(0xe0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f));
            } else {
                out.push(0xf0 | (cp >> 18), 0x80 | ((cp >> 12) & 0x3f),
                    0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f));
            }
        }
        return new Uint8Array(out);
    }
}

if (typeof globalThis.TextDecoder === 'undefined') globalThis.TextDecoder = Utf8Decoder;
if (typeof globalThis.TextEncoder === 'undefined') globalThis.TextEncoder = Utf8Encoder;
