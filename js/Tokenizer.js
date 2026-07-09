export class Tokenizer {
    constructor() {
        this.vocabByText = new Map(); // token text (utf-8 decoded) -> id
        this.vocabByLatin1 = new Map(); // raw latin1 fallback -> id
        this.idToText = [];
        this.idToBytes = [];
        this.merges = new Map(); // `left,right` -> { rank, mergedId }
        this.mergeEnabled = false;
        this.decoder = new TextDecoder("utf-8");
        this.encoder = new TextEncoder();
        this.bpePattern = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
    }

    // vocab.json and vocab.bin are both supported.
    // pass vocabUrl, mergesUrl.
    async load(vocabUrl, mergesUrl = null) {
        try {
            const isBin = /\.bin$/i.test(vocabUrl);
            if (isBin) await this.loadFromBinary(vocabUrl);
            else await this.loadFromJson(vocabUrl);

            if (mergesUrl) await this.loadMerges(mergesUrl);
        } catch (err) {
            console.error("Tokenizer load error:", err);
            throw err;
        }
    }

    resetVocab() {
        this.vocabByText.clear();
        this.vocabByLatin1.clear();
        this.idToText = [];
        this.idToBytes = [];
        this.merges.clear();
        this.mergeEnabled = false;
    }

    async loadFromJson(vocabUrl) {
        const res = await fetch(vocabUrl);
        if (!res.ok) throw new Error(`Failed to fetch vocab: ${res.statusText}`);
        const vocabJson = await res.json();

        this.resetVocab();
        for (const [text, id] of Object.entries(vocabJson)) {
            const bytes = this.encoder.encode(text);
            this.vocabByText.set(text, id);
            this.vocabByLatin1.set(this.bytesToLatin1(bytes), id);
            this.idToText[id] = text;
            this.idToBytes[id] = bytes;
        }
        console.log(`Loaded ${Object.keys(vocabJson).length} tokens from ${vocabUrl}`);
    }

    async loadFromBinary(vocabUrl) {
        const res = await fetch(vocabUrl);
        if (!res.ok) throw new Error(`Failed to fetch vocab.bin: ${res.statusText}`);

        const buf = new Uint8Array(await res.arrayBuffer());
        if (buf.length < 4) throw new Error("Invalid vocab.bin");
        const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
        const size = dv.getInt32(0, true);

        this.resetVocab();
        let off = 4;
        for (let i = 0; i < size; i++) {
            if (off + 4 > buf.length) throw new Error(`vocab.bin truncated at token ${i}`);
            const len = dv.getInt32(off, true);
            off += 4;
            if (len < 0 || off + len > buf.length) throw new Error(`vocab.bin malformed at token ${i}`);
            const tokenBytes = buf.slice(off, off + len);
            off += len;

            const text = this.decoder.decode(tokenBytes);
            this.vocabByText.set(text, i);
            this.vocabByLatin1.set(this.bytesToLatin1(tokenBytes), i);
            this.idToText[i] = text;
            this.idToBytes[i] = tokenBytes;
        }
        console.log(`Loaded ${size} tokens from ${vocabUrl}`);
    }

    async loadMerges(mergesUrl) {
        const res = await fetch(mergesUrl);
        if (!res.ok) throw new Error(`Failed to fetch merges: ${res.statusText}`);
        const text = await res.text();

        let rank = 0;
        this.merges.clear();
        for (const rawLine of text.split(/\r?\n/)) {
            const line = rawLine.trim();
            if (!line || line.startsWith("#")) continue;
            const parts = line.split(/\s+/);
            if (parts.length < 2) continue;

            const leftToken = this.normalizeMergeToken(parts[0]);
            const rightToken = this.normalizeMergeToken(parts[1]);
            const leftId = this.findTokenByText(leftToken);
            const rightId = this.findTokenByText(rightToken);
            if (leftId < 0 || rightId < 0) continue;

            const mergedText = `${this.idToText[leftId]}${this.idToText[rightId]}`;
            const mergedId = this.vocabByText.get(mergedText);
            if (mergedId === undefined) continue;

            const key = this.pairKey(leftId, rightId);
            if (!this.merges.has(key)) {
                this.merges.set(key, { rank: rank++, mergedId });
            }
        }

        this.mergeEnabled = this.merges.size > 0;
        console.log(`Loaded ${this.merges.size} merge rules from ${mergesUrl}`);
    }

    bytesToLatin1(bytes) {
        let out = "";
        for (let i = 0; i < bytes.length; i++) out += String.fromCharCode(bytes[i]);
        return out;
    }

    utf8Len(byte) {
        if (byte < 0x80) return 1;
        if ((byte & 0xe0) === 0xc0) return 2;
        if ((byte & 0xf0) === 0xe0) return 3;
        if ((byte & 0xf8) === 0xf0) return 4;
        return 1;
    }

    pairKey(left, right) {
        return `${left},${right}`;
    }

    normalizeMergeToken(token) {
        if (token.startsWith("Ġ")) return ` ${token.slice(1)}`;
        return token;
    }

    findTokenByText(text) {
        let id = this.vocabByText.get(text);
        if (id !== undefined) return id;
        id = this.vocabByLatin1.get(text);
        return id === undefined ? -1 : id;
    }

    findTokenByBytes(bytes) {
        const text = this.decoder.decode(bytes);
        let id = this.vocabByText.get(text);
        if (id !== undefined) return id;
        id = this.vocabByLatin1.get(this.bytesToLatin1(bytes));
        return id === undefined ? -1 : id;
    }

    // Decode a single token ID to its text representation.
    decodeToken(tokenId) {
        const bytes = this.idToBytes[tokenId];
        if (!bytes) return "";
        return this.decoder.decode(bytes).replace(/Ġ/g, " ");
    }

    // Decode a token-id array.
    decode(tokenIds) {
        return tokenIds.map((id) => this.decodeToken(id)).join("");
    }

    encodeBpeWord(word, maxTokens) {
        const bytes = this.encoder.encode(word);
        if (bytes.length === 0 || maxTokens <= 0) return [];

        const symbols = [];
        for (let pos = 0; pos < bytes.length;) {
            let clen = this.utf8Len(bytes[pos]);
            if (pos + clen > bytes.length) clen = bytes.length - pos;
            const symbolBytes = bytes.slice(pos, pos + clen);
            const id = this.findTokenByBytes(symbolBytes);
            if (id >= 0) {
                symbols.push(id);
            } else {
                for (let i = 0; i < clen; i++) {
                    const byteId = this.findTokenByBytes(Uint8Array.of(bytes[pos + i]));
                    if (byteId >= 0) symbols.push(byteId);
                }
            }
            pos += clen;
        }

        while (symbols.length > 1) {
            let bestIdx = -1;
            let bestRank = Number.MAX_SAFE_INTEGER;
            let bestMerged = -1;

            for (let i = 0; i < symbols.length - 1; i++) {
                const merge = this.merges.get(this.pairKey(symbols[i], symbols[i + 1]));
                if (!merge) continue;
                if (merge.rank >= bestRank) continue;
                if (merge.mergedId < 0) continue;

                bestIdx = i;
                bestRank = merge.rank;
                bestMerged = merge.mergedId;
            }

            if (bestIdx < 0) break;
            symbols[bestIdx] = bestMerged;
            symbols.splice(bestIdx + 1, 1);
        }

        return symbols.slice(0, maxTokens);
    }

    encodeGreedy(text, maxTokens) {
        const bytes = this.encoder.encode(text);
        const tokens = [];
        let pos = 0;

        while (pos < bytes.length && tokens.length < maxTokens) {
            let bestId = -1;
            let bestLen = 0;

            for (let i = 0; i < this.idToBytes.length; i++) {
                const tokenBytes = this.idToBytes[i];
                if (!tokenBytes || tokenBytes.length <= bestLen || pos + tokenBytes.length > bytes.length) continue;
                let match = true;
                for (let j = 0; j < tokenBytes.length; j++) {
                    if (bytes[pos + j] !== tokenBytes[j]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    bestId = i;
                    bestLen = tokenBytes.length;
                }
            }

            if (bestId < 0) {
                pos++;
                continue;
            }

            tokens.push(bestId);
            pos += bestLen;
        }

        return tokens;
    }

    // Tokenize with GPT-2-style regex pretokenization and run one BPE pass per chunk.
    encode(text, maxTokens = 256) {
        if (maxTokens <= 0) return [];
        if (this.vocabByText.size === 0 && this.vocabByLatin1.size === 0) return [];
        if (!this.mergeEnabled) return this.encodeGreedy(text, maxTokens);

        const tokens = [];
        for (const match of text.matchAll(this.bpePattern)) {
            const chunk = match[0];
            if (!chunk || tokens.length >= maxTokens) continue;

            const remain = maxTokens - tokens.length;
            if (remain <= 0) break;
            const part = this.encodeBpeWord(chunk, remain);
            for (const id of part) tokens.push(id);
        }

        return tokens;
    }
}
