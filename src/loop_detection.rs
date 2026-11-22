//! Official `crisperwhisper.loop_detection` — post-hoc n-gram loop scan.

/// Per-ngram-size repetition thresholds for `generate_with_repair`.
pub fn default_repair_thresholds() -> Vec<(usize, usize)> {
    let mut t = vec![(1, 8), (2, 8), (3, 4), (4, 3), (5, 3)];
    for n in 6..25 {
        t.push((n, 3));
    }
    t
}

/// Find the earliest position where consecutive n-grams repeat.
/// Returns `(loop_start_index, ngram)`.
pub fn find_token_loop(ids: &[u32], thresholds: &[(usize, usize)]) -> Option<(usize, Vec<u32>)> {
    for i in 0..ids.len() {
        for &(n, needed_reps) in thresholds {
            let needed = n * needed_reps;
            if i + needed > ids.len() {
                continue;
            }
            let gram = &ids[i..i + n];
            let mut ok = true;
            for r in 1..needed_reps {
                if &ids[i + r * n..i + (r + 1) * n] != gram {
                    ok = false;
                    break;
                }
            }
            if ok {
                return Some((i, gram.to_vec()));
            }
        }
    }
    None
}

pub fn final_trim(generated: &[u32], thresholds: &[(usize, usize)], keep_reps: usize) -> Vec<u32> {
    match find_token_loop(generated, thresholds) {
        None => generated.to_vec(),
        Some((loop_start, gram)) => generated[..loop_start + gram.len() * keep_reps].to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_repeated_trigram() {
        // 3-gram repeated 4 times starting at 0
        let gram = [1u32, 2, 3];
        let mut ids = Vec::new();
        for _ in 0..4 {
            ids.extend_from_slice(&gram);
        }
        let hit = find_token_loop(&ids, &default_repair_thresholds()).unwrap();
        assert_eq!(hit.0, 0);
        assert_eq!(hit.1, vec![1, 2, 3]);
    }
}
