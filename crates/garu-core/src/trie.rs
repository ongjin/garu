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

/// Internal trie node.
#[derive(Debug, Clone)]
struct TrieNode {
    children: Vec<(char, usize)>,
    entries: Vec<DictEntry>,
}

impl TrieNode {
    fn new() -> Self {
        TrieNode {
            children: Vec::new(),
            entries: Vec::new(),
        }
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

/// Trie-based dictionary for morphological lookup.
#[derive(Debug, Clone)]
pub struct Dict {
    nodes: Vec<TrieNode>,
    entry_count: usize,
}

const MAGIC: &[u8; 4] = b"GARU";
const VERSION: u32 = 1;

impl Dict {
    /// Create an empty dictionary.
    pub fn new() -> Self {
        Dict {
            nodes: vec![TrieNode::new()], // root node
            entry_count: 0,
        }
    }

    /// Insert a word with its dictionary entry.
    pub fn insert(&mut self, word: &str, entry: DictEntry) {
        let mut node_idx = 0;
        for c in word.chars() {
            let next_id = self.nodes.len();
            let (child_idx, created) = self.nodes[node_idx].get_or_create_child(c, next_id);
            if created {
                self.nodes.push(TrieNode::new());
            }
            node_idx = child_idx;
        }
        self.nodes[node_idx].entries.push(entry);
        self.entry_count += 1;
    }

    /// Look up an exact word. Returns the entries for that word (empty slice if not found).
    pub fn lookup(&self, word: &str) -> &[DictEntry] {
        let mut node_idx = 0;
        for c in word.chars() {
            match self.nodes[node_idx].get_child(c) {
                Some(idx) => node_idx = idx,
                None => return &[],
            }
        }
        &self.nodes[node_idx].entries
    }

    /// Find all prefixes of `text` that exist in the dictionary.
    /// Returns vec of (byte_length_of_prefix, entries).
    pub fn common_prefix_search(&self, text: &str) -> Vec<(usize, &[DictEntry])> {
        let mut results = Vec::new();
        let mut node_idx = 0;
        let mut byte_pos = 0;

        for c in text.chars() {
            match self.nodes[node_idx].get_child(c) {
                Some(idx) => {
                    node_idx = idx;
                    byte_pos += c.len_utf8();
                    let entries = &self.nodes[node_idx].entries;
                    if !entries.is_empty() {
                        results.push((byte_pos, entries.as_slice()));
                    }
                }
                None => break,
            }
        }

        results
    }

    /// Total number of entries inserted.
    pub fn len(&self) -> usize {
        self.entry_count
    }

    /// Whether the dictionary is empty.
    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    /// Serialize the dictionary to bytes.
    ///
    /// Format: "GARU" magic + u32 version(1) + u32 entry_count + entries
    /// Each entry: u16 word_len + word_bytes + u8 num_entries + for each entry:
    ///   u8 num_morphemes + morphemes + f32 score
    /// Each morpheme: u16 text_len + text_bytes + u8 pos_byte
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());

        // Collect all words and their entries via DFS
        let mut word_entries: Vec<(String, &[DictEntry])> = Vec::new();
        self.collect_entries(0, &mut String::new(), &mut word_entries);

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

    /// Deserialize a dictionary from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let mut pos = 0;

        // Magic
        if data.len() < 12 {
            return Err("Data too short for header".into());
        }
        if &data[0..4] != MAGIC {
            return Err("Invalid magic bytes".into());
        }
        pos += 4;

        // Version
        let version = u32::from_le_bytes(
            data[pos..pos + 4].try_into().map_err(|_| "Bad version bytes")?
        );
        if version != VERSION {
            return Err(format!("Unsupported version: {}", version));
        }
        pos += 4;

        // Entry count
        let word_count = u32::from_le_bytes(
            data[pos..pos + 4].try_into().map_err(|_| "Bad entry count bytes")?
        ) as usize;
        pos += 4;

        let mut dict = Dict::new();

        for _ in 0..word_count {
            // Word
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

            // Number of entries for this word
            if pos >= data.len() {
                return Err("Unexpected end of data reading num_entries".into());
            }
            let num_entries = data[pos] as usize;
            pos += 1;

            for _ in 0..num_entries {
                // Number of morphemes
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

                    if pos_byte > 38 {
                        return Err(format!("Invalid POS byte: {}", pos_byte));
                    }
                    // Safety: pos_byte is validated to be in range 0..=38,
                    // matching the 39 variants of Pos (#[repr(u8)]).
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

    /// DFS helper to collect all words and their entries from the trie.
    fn collect_entries<'a>(
        &'a self,
        node_idx: usize,
        prefix: &mut String,
        results: &mut Vec<(String, &'a [DictEntry])>,
    ) {
        let node = &self.nodes[node_idx];
        if !node.entries.is_empty() {
            results.push((prefix.clone(), node.entries.as_slice()));
        }
        // Sort children for deterministic serialization order
        let mut children: Vec<_> = node.children.clone();
        children.sort_by_key(|(c, _)| *c);
        for (c, child_idx) in children {
            prefix.push(c);
            self.collect_entries(child_idx, prefix, results);
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
