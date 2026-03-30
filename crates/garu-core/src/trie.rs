use crate::types::Pos;

/// A single morpheme: surface form + POS tag.
#[derive(Debug, Clone, PartialEq)]
pub struct Morpheme {
    pub text: String,
    pub pos: Pos,
}

/// A dictionary entry: one or more morphemes with a score.
#[derive(Debug, Clone, PartialEq)]
pub struct DictEntry {
    pub morphemes: Vec<Morpheme>,
    pub score: f32,
}

// ---- Trie internals (kept for v1 backward compat) ----

#[derive(Debug, Clone)]
struct TrieNode {
    children: Vec<(char, usize)>,
    entries: Vec<DictEntry>,
}

impl TrieNode {
    fn new() -> Self {
        TrieNode { children: Vec::new(), entries: Vec::new() }
    }
    fn get_child(&self, c: char) -> Option<usize> {
        self.children.iter().find(|(ch, _)| *ch == c).map(|(_, idx)| *idx)
    }
    fn get_or_create_child(&mut self, c: char, next_id: usize) -> (usize, bool) {
        if let Some(idx) = self.get_child(c) {
            (idx, false)
        } else {
            self.children.push((c, next_id));
            (next_id, true)
        }
    }
}

// ---- FST value decoding ----

/// Decode FST value returning all POS entries (primary + optional secondary).
fn decode_fst_value_multi(value: u64) -> Vec<(Pos, f32)> {
    let pos1_byte = (value & 0xFF) as u8;
    let qfreq1 = ((value >> 8) & 0xFFFF) as u16;
    let pos2_byte = ((value >> 24) & 0xFF) as u8;
    let qfreq2 = ((value >> 32) & 0xFFFF) as u16;

    let mut results = Vec::with_capacity(2);

    // Primary
    if pos1_byte <= 41 {
        let pos: Pos = unsafe { std::mem::transmute(pos1_byte) };
        let score = if qfreq1 > 0 { -(qfreq1 as f32 / 65535.0).ln() } else { 15.0 };
        results.push((pos, score));
    }

    // Secondary (if present: pos2_byte != 0xFF and qfreq2 > 0)
    if pos2_byte <= 41 && qfreq2 > 0 {
        let pos: Pos = unsafe { std::mem::transmute(pos2_byte) };
        let score = if qfreq2 > 0 { -(qfreq2 as f32 / 65535.0).ln() } else { 15.0 };
        results.push((pos, score));
    }

    results
}

// ---- Dict with dual backend ----

const MAGIC: &[u8; 4] = b"GARU";

enum DictBackend {
    Trie {
        nodes: Vec<TrieNode>,
        entry_count: usize,
    },
    Fst {
        map: fst::Map<Vec<u8>>,
    },
}

/// Dictionary for morphological lookup. Supports Trie (v1) and FST (v2) backends.
pub struct Dict {
    backend: DictBackend,
}

impl Dict {
    /// Create an empty Trie-based dictionary.
    pub fn new() -> Self {
        Dict {
            backend: DictBackend::Trie {
                nodes: vec![TrieNode::new()],
                entry_count: 0,
            },
        }
    }

    /// Insert a word (Trie backend only, panics on FST).
    pub fn insert(&mut self, word: &str, entry: DictEntry) {
        match &mut self.backend {
            DictBackend::Trie { nodes, entry_count } => {
                let mut node_idx = 0;
                for c in word.chars() {
                    let next_id = nodes.len();
                    let (child_idx, created) = nodes[node_idx].get_or_create_child(c, next_id);
                    if created {
                        nodes.push(TrieNode::new());
                    }
                    node_idx = child_idx;
                }
                nodes[node_idx].entries.push(entry);
                *entry_count += 1;
            }
            DictBackend::Fst { .. } => panic!("Cannot insert into FST dict"),
        }
    }

    /// Look up an exact word.
    pub fn lookup(&self, word: &str) -> Vec<DictEntry> {
        match &self.backend {
            DictBackend::Trie { nodes, .. } => {
                let mut node_idx = 0;
                for c in word.chars() {
                    match nodes[node_idx].get_child(c) {
                        Some(idx) => node_idx = idx,
                        None => return vec![],
                    }
                }
                nodes[node_idx].entries.clone()
            }
            DictBackend::Fst { map } => {
                match map.get(word.as_bytes()) {
                    Some(value) => {
                        decode_fst_value_multi(value).into_iter().map(|(pos, score)| {
                            DictEntry {
                                morphemes: vec![Morpheme { text: word.to_string(), pos }],
                                score,
                            }
                        }).collect()
                    }
                    None => vec![],
                }
            }
        }
    }

    /// Find all prefixes of `text` that exist in the dictionary.
    /// Returns vec of (byte_length_of_prefix, entries).
    pub fn common_prefix_search(&self, text: &str) -> Vec<(usize, Vec<DictEntry>)> {
        match &self.backend {
            DictBackend::Trie { nodes, .. } => {
                let mut results = Vec::new();
                let mut node_idx = 0;
                let mut byte_pos = 0;
                for c in text.chars() {
                    match nodes[node_idx].get_child(c) {
                        Some(idx) => {
                            node_idx = idx;
                            byte_pos += c.len_utf8();
                            let entries = &nodes[node_idx].entries;
                            if !entries.is_empty() {
                                results.push((byte_pos, entries.clone()));
                            }
                        }
                        None => break,
                    }
                }
                results
            }
            DictBackend::Fst { map } => {
                let bytes = text.as_bytes();
                let mut results = Vec::new();
                let mut byte_pos = 0;
                for ch in text.chars() {
                    byte_pos += ch.len_utf8();
                    if let Some(value) = map.get(&bytes[..byte_pos]) {
                        let surface = std::str::from_utf8(&bytes[..byte_pos]).unwrap_or("");
                        let entries: Vec<DictEntry> = decode_fst_value_multi(value)
                            .into_iter()
                            .map(|(pos, score)| DictEntry {
                                morphemes: vec![Morpheme { text: surface.to_string(), pos }],
                                score,
                            })
                            .collect();
                        if !entries.is_empty() {
                            results.push((byte_pos, entries));
                        }
                    }
                }
                results
            }
        }
    }

    /// Total number of entries.
    pub fn len(&self) -> usize {
        match &self.backend {
            DictBackend::Trie { entry_count, .. } => *entry_count,
            DictBackend::Fst { map } => map.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Serialize (Trie backend only).
    pub fn to_bytes(&self) -> Vec<u8> {
        match &self.backend {
            DictBackend::Trie { nodes, .. } => {
                let mut buf = Vec::new();
                buf.extend_from_slice(MAGIC);
                buf.extend_from_slice(&1u32.to_le_bytes()); // version 1

                let mut word_entries: Vec<(String, &[DictEntry])> = Vec::new();
                Self::collect_entries_static(nodes, 0, &mut String::new(), &mut word_entries);

                buf.extend_from_slice(&(word_entries.len() as u32).to_le_bytes());

                for (word, entries) in &word_entries {
                    let word_bytes = word.as_bytes();
                    buf.extend_from_slice(&(word_bytes.len() as u16).to_le_bytes());
                    buf.extend_from_slice(word_bytes);
                    buf.push(entries.len() as u8);
                    for entry in *entries {
                        buf.push(entry.morphemes.len() as u8);
                        for morpheme in &entry.morphemes {
                            let text_bytes = morpheme.text.as_bytes();
                            buf.extend_from_slice(&(text_bytes.len() as u16).to_le_bytes());
                            buf.extend_from_slice(text_bytes);
                            buf.push(morpheme.pos as u8);
                        }
                        buf.extend_from_slice(&entry.score.to_le_bytes());
                    }
                }
                buf
            }
            DictBackend::Fst { .. } => panic!("Use FST bytes directly, not to_bytes()"),
        }
    }

    /// Deserialize a dictionary from bytes. Supports v1 (Trie) and v2 (FST).
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 8 {
            return Err("Data too short for header".into());
        }
        if &data[0..4] != MAGIC {
            return Err("Invalid magic bytes".into());
        }
        let version = u32::from_le_bytes(
            data[4..8].try_into().map_err(|_| "Bad version bytes")?
        );

        match version {
            1 => Self::from_bytes_v1(data),
            2 => Self::from_bytes_v2(data),
            _ => Err(format!("Unsupported dict version: {}", version)),
        }
    }

    /// Parse v1 Trie format.
    fn from_bytes_v1(data: &[u8]) -> Result<Self, String> {
        let mut pos = 8; // skip magic + version

        let word_count = u32::from_le_bytes(
            data[pos..pos + 4].try_into().map_err(|_| "Bad entry count bytes")?
        ) as usize;
        pos += 4;

        let mut dict = Dict::new();

        for _ in 0..word_count {
            if pos + 2 > data.len() {
                return Err("Unexpected end of data reading word length".into());
            }
            let word_len = u16::from_le_bytes(
                data[pos..pos + 2].try_into().map_err(|_| "Bad word len")?
            ) as usize;
            pos += 2;

            if pos + word_len > data.len() {
                return Err("Unexpected end of data reading word".into());
            }
            let word = std::str::from_utf8(&data[pos..pos + word_len])
                .map_err(|e| format!("Invalid UTF-8 in word: {}", e))?;
            pos += word_len;

            if pos >= data.len() {
                return Err("Unexpected end of data reading num_entries".into());
            }
            let num_entries = data[pos] as usize;
            pos += 1;

            for _ in 0..num_entries {
                if pos >= data.len() {
                    return Err("Unexpected end of data reading num_morphemes".into());
                }
                let num_morphemes = data[pos] as usize;
                pos += 1;

                let mut morphemes = Vec::with_capacity(num_morphemes);
                for _ in 0..num_morphemes {
                    if pos + 2 > data.len() {
                        return Err("Unexpected end of data reading morpheme text length".into());
                    }
                    let text_len = u16::from_le_bytes(
                        data[pos..pos + 2].try_into().map_err(|_| "Bad morpheme text len")?
                    ) as usize;
                    pos += 2;

                    if pos + text_len > data.len() {
                        return Err("Unexpected end of data reading morpheme text".into());
                    }
                    let text = std::str::from_utf8(&data[pos..pos + text_len])
                        .map_err(|e| format!("Invalid UTF-8 in morpheme: {}", e))?
                        .to_string();
                    pos += text_len;

                    if pos >= data.len() {
                        return Err("Unexpected end of data reading pos byte".into());
                    }
                    let pos_byte = data[pos];
                    pos += 1;

                    if pos_byte > 41 {
                        return Err(format!("Invalid POS byte: {}", pos_byte));
                    }
                    let pos_tag: Pos = unsafe { std::mem::transmute(pos_byte) };
                    morphemes.push(Morpheme { text, pos: pos_tag });
                }

                if pos + 4 > data.len() {
                    return Err("Unexpected end of data reading score".into());
                }
                let score = f32::from_le_bytes(
                    data[pos..pos + 4].try_into().map_err(|_| "Bad score bytes")?
                );
                pos += 4;

                dict.insert(word, DictEntry { morphemes, score });
            }
        }

        Ok(dict)
    }

    /// Parse v2 FST format.
    /// Format: "GARU" + u32 version(2) + u32 fst_len + [fst bytes]
    /// The FST maps word_bytes -> pos_byte as u64.
    fn from_bytes_v2(data: &[u8]) -> Result<Self, String> {
        if data.len() < 12 {
            return Err("FST dict data too short".into());
        }
        let fst_len = u32::from_le_bytes(
            data[8..12].try_into().map_err(|_| "Bad FST length")?
        ) as usize;

        if 12 + fst_len > data.len() {
            return Err("FST data truncated".into());
        }

        let fst_data = data[12..12 + fst_len].to_vec();
        let map = fst::Map::new(fst_data)
            .map_err(|e| format!("Invalid FST data: {}", e))?;

        Ok(Dict {
            backend: DictBackend::Fst { map },
        })
    }

    /// DFS helper to collect entries from trie nodes.
    fn collect_entries_static<'a>(
        nodes: &'a [TrieNode],
        node_idx: usize,
        prefix: &mut String,
        results: &mut Vec<(String, &'a [DictEntry])>,
    ) {
        let node = &nodes[node_idx];
        if !node.entries.is_empty() {
            results.push((prefix.clone(), node.entries.as_slice()));
        }
        let mut children: Vec<_> = node.children.clone();
        children.sort_by_key(|(c, _)| *c);
        for (c, child_idx) in children {
            prefix.push(c);
            Self::collect_entries_static(nodes, child_idx, prefix, results);
            prefix.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Pos;

    fn make_entry(text: &str, pos: Pos, score: f32) -> DictEntry {
        DictEntry {
            morphemes: vec![Morpheme {
                text: text.to_string(),
                pos,
            }],
            score,
        }
    }

    fn make_compound_entry(parts: &[(&str, Pos)], score: f32) -> DictEntry {
        DictEntry {
            morphemes: parts
                .iter()
                .map(|(t, p)| Morpheme {
                    text: t.to_string(),
                    pos: *p,
                })
                .collect(),
            score,
        }
    }

    #[test]
    fn test_insert_and_lookup() {
        let mut dict = Dict::new();
        dict.insert("나", make_entry("나", Pos::NP, -2.0));
        dict.insert("나라", make_entry("나라", Pos::NNG, -3.5));

        let entries = dict.lookup("나");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].morphemes[0].text, "나");
        assert_eq!(entries[0].morphemes[0].pos, Pos::NP);

        let entries = dict.lookup("나라");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].morphemes[0].text, "나라");

        // Not found
        let entries = dict.lookup("없음");
        assert!(entries.is_empty());

        assert_eq!(dict.len(), 2);
        assert!(!dict.is_empty());
    }

    #[test]
    fn test_common_prefix_search() {
        let mut dict = Dict::new();
        dict.insert("나", make_entry("나", Pos::NP, -2.0));
        dict.insert("나라", make_entry("나라", Pos::NNG, -3.5));
        dict.insert("나라꽃", make_entry("나라꽃", Pos::NNG, -5.0));

        let results = dict.common_prefix_search("나라꽃이");
        assert_eq!(results.len(), 3);

        // "나" = 3 bytes
        assert_eq!(results[0].0, 3);
        assert_eq!(results[0].1[0].morphemes[0].text, "나");

        // "나라" = 6 bytes
        assert_eq!(results[1].0, 6);
        assert_eq!(results[1].1[0].morphemes[0].text, "나라");

        // "나라꽃" = 9 bytes
        assert_eq!(results[2].0, 9);
        assert_eq!(results[2].1[0].morphemes[0].text, "나라꽃");
    }

    #[test]
    fn test_serialize_roundtrip() {
        let mut dict = Dict::new();
        dict.insert("나", make_entry("나", Pos::NP, -2.0));
        dict.insert("나라", make_entry("나라", Pos::NNG, -3.5));
        dict.insert(
            "학교",
            make_compound_entry(&[("학교", Pos::NNG)], -4.0),
        );

        let bytes = dict.to_bytes();
        let dict2 = Dict::from_bytes(&bytes).expect("deserialization failed");

        assert_eq!(dict2.len(), dict.len());

        let e1 = dict2.lookup("나");
        assert_eq!(e1.len(), 1);
        assert_eq!(e1[0].morphemes[0].pos, Pos::NP);
        assert!((e1[0].score - (-2.0)).abs() < f32::EPSILON);

        let e2 = dict2.lookup("나라");
        assert_eq!(e2.len(), 1);
        assert_eq!(e2[0].morphemes[0].pos, Pos::NNG);

        let e3 = dict2.lookup("학교");
        assert_eq!(e3.len(), 1);
        assert_eq!(e3[0].morphemes[0].text, "학교");
    }

    #[test]
    fn test_empty_dict() {
        let dict = Dict::new();
        assert!(dict.is_empty());
        assert_eq!(dict.len(), 0);
        assert!(dict.lookup("아무거나").is_empty());
        assert!(dict.common_prefix_search("테스트").is_empty());

        // Roundtrip empty
        let bytes = dict.to_bytes();
        let dict2 = Dict::from_bytes(&bytes).expect("deserialization failed");
        assert!(dict2.is_empty());
    }

    #[test]
    fn test_invalid_magic() {
        let result = Dict::from_bytes(b"BAAD\x01\x00\x00\x00\x00\x00\x00\x00");
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_entries_same_word() {
        let mut dict = Dict::new();
        dict.insert("나", make_entry("나", Pos::NP, -2.0));
        dict.insert("나", make_entry("나", Pos::VV, -4.0));

        let entries = dict.lookup("나");
        assert_eq!(entries.len(), 2);
        assert_eq!(dict.len(), 2);

        // Roundtrip
        let bytes = dict.to_bytes();
        let dict2 = Dict::from_bytes(&bytes).expect("deserialization failed");
        let entries2 = dict2.lookup("나");
        assert_eq!(entries2.len(), 2);
    }
}
