/// CRF (Conditional Random Field) layer with Viterbi decoding.
///
/// Given per-timestep emission scores and a transition matrix, the decoder
/// finds the tag sequence that maximises the total score.

/// CRF decoder holding a transition matrix.
pub struct Crf {
    /// transitions\[i\]\[j\] = score for moving from tag i to tag j.
    pub transitions: Vec<Vec<f32>>,
    /// Number of tags (rows/cols of the transition matrix).
    pub num_tags: usize,
}

impl Crf {
    /// Create a new CRF from a square transition matrix.
    ///
    /// # Panics
    /// Panics if `transitions` is not `num_tags x num_tags`.
    pub fn new(transitions: Vec<Vec<f32>>, num_tags: usize) -> Self {
        assert_eq!(transitions.len(), num_tags);
        for row in &transitions {
            assert_eq!(row.len(), num_tags);
        }
        Self {
            transitions,
            num_tags,
        }
    }

    /// Viterbi decode: find the single best tag sequence.
    ///
    /// Returns `(best_path, best_score)`.
    /// For empty emissions returns `(vec![], 0.0)`.
    pub fn decode(&self, emissions: &[Vec<f32>]) -> (Vec<usize>, f32) {
        let t_len = emissions.len();
        if t_len == 0 {
            return (vec![], 0.0);
        }
        let n = self.num_tags;

        // viterbi[t][j] = best score ending in tag j at time t
        let mut viterbi = vec![vec![0.0_f32; n]; t_len];
        // backpointer[t][j] = best previous tag for tag j at time t
        let mut backptr = vec![vec![0usize; n]; t_len];

        // Initialisation
        for j in 0..n {
            viterbi[0][j] = emissions[0][j];
        }

        // Forward pass
        for t in 1..t_len {
            for j in 0..n {
                let mut best_score = f32::NEG_INFINITY;
                let mut best_prev = 0usize;
                for i in 0..n {
                    let score = viterbi[t - 1][i] + self.transitions[i][j];
                    if score > best_score {
                        best_score = score;
                        best_prev = i;
                    }
                }
                viterbi[t][j] = best_score + emissions[t][j];
                backptr[t][j] = best_prev;
            }
        }

        // Find best final tag
        let mut best_tag = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for j in 0..n {
            if viterbi[t_len - 1][j] > best_score {
                best_score = viterbi[t_len - 1][j];
                best_tag = j;
            }
        }

        // Backtrack
        let mut path = vec![0usize; t_len];
        path[t_len - 1] = best_tag;
        for t in (1..t_len).rev() {
            path[t - 1] = backptr[t][path[t]];
        }

        (path, best_score)
    }

    /// Beam-search decode returning up to `n` best tag sequences.
    ///
    /// Results are sorted by score in descending order.
    /// For `n == 1` this delegates to [`Self::decode`].
    pub fn decode_topn(
        &self,
        emissions: &[Vec<f32>],
        n: usize,
    ) -> Vec<(Vec<usize>, f32)> {
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            let result = self.decode(emissions);
            return vec![result];
        }

        let t_len = emissions.len();
        if t_len == 0 {
            return vec![(vec![], 0.0)];
        }

        let num = self.num_tags;

        // Each beam entry: (path_so_far, score)
        // Initialise with single-element paths at t=0
        let mut beam: Vec<(Vec<usize>, f32)> = (0..num)
            .map(|j| (vec![j], emissions[0][j]))
            .collect();

        // Keep only top-n
        beam.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        beam.truncate(n);

        // Expand at each subsequent timestep
        for t in 1..t_len {
            let mut next_beam: Vec<(Vec<usize>, f32)> =
                Vec::with_capacity(beam.len() * num);

            for (path, score) in &beam {
                let prev_tag = *path.last().unwrap();
                for j in 0..num {
                    let new_score =
                        score + self.transitions[prev_tag][j] + emissions[t][j];
                    let mut new_path = path.clone();
                    new_path.push(j);
                    next_beam.push((new_path, new_score));
                }
            }

            next_beam.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            next_beam.truncate(n);
            beam = next_beam;
        }

        beam.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        beam
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple 2-tag, 3-step: verify shapes and finite score.
    #[test]
    fn test_simple_2tag_3step() {
        let transitions = vec![vec![0.5, 0.1], vec![0.2, 0.3]];
        let crf = Crf::new(transitions, 2);

        let emissions = vec![vec![1.0, 0.5], vec![0.3, 0.8], vec![0.6, 0.4]];
        let (path, score) = crf.decode(&emissions);

        assert_eq!(path.len(), 3);
        assert!(score.is_finite());
        // All tags should be in range [0, 2)
        for &tag in &path {
            assert!(tag < 2);
        }
    }

    /// Strong transition tag0->tag0 should pin all tags to 0 despite
    /// emissions favouring tag1.
    #[test]
    fn test_strong_transition_pins_tag0() {
        // Huge self-loop score for tag0, penalty everywhere else
        let transitions = vec![vec![10.0, -10.0], vec![-10.0, 0.0]];
        let crf = Crf::new(transitions, 2);

        // Emissions slightly favour tag1 at every step
        let emissions = vec![
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
        ];
        let (path, _score) = crf.decode(&emissions);

        assert_eq!(path, vec![0, 0, 0, 0]);
    }

    /// Empty emissions should return ([], 0.0).
    #[test]
    fn test_empty_emissions() {
        let transitions = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let crf = Crf::new(transitions, 2);

        let (path, score) = crf.decode(&[]);
        assert!(path.is_empty());
        assert_eq!(score, 0.0);
    }

    /// Top-N: results sorted descending, count <= possible paths.
    #[test]
    fn test_topn_sorted_and_bounded() {
        let transitions = vec![vec![0.5, 0.1], vec![0.2, 0.3]];
        let crf = Crf::new(transitions, 2);

        let emissions = vec![vec![1.0, 0.5], vec![0.3, 0.8], vec![0.6, 0.4]];
        let results = crf.decode_topn(&emissions, 5);

        // 2 tags, 3 steps => 2^3 = 8 possible paths, requested 5
        assert!(results.len() <= 5);
        assert!(!results.is_empty());

        // Verify descending score order
        for w in results.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "scores not descending: {} < {}",
                w[0].1,
                w[1].1
            );
        }

        // All paths should have length 3
        for (path, score) in &results {
            assert_eq!(path.len(), 3);
            assert!(score.is_finite());
        }
    }

    /// Top-1 should match decode exactly.
    #[test]
    fn test_topn_1_matches_decode() {
        let transitions = vec![vec![0.5, 0.1], vec![0.2, 0.3]];
        let crf = Crf::new(transitions, 2);

        let emissions = vec![vec![1.0, 0.5], vec![0.3, 0.8]];
        let (path, score) = crf.decode(&emissions);
        let topn = crf.decode_topn(&emissions, 1);

        assert_eq!(topn.len(), 1);
        assert_eq!(topn[0].0, path);
        assert_eq!(topn[0].1, score);
    }
}
