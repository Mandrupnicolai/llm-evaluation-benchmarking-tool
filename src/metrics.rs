use std::collections::{HashMap, HashSet};

pub fn compute_metrics(
    metric_names: &[String],
    output: &str,
    reference: &str,
) -> HashMap<String, f64> {
    let mut scores = HashMap::new();
    let norm_output = normalize(output);
    let norm_ref = normalize(reference);

    for name in metric_names {
        let score = match name.as_str() {
            "exact_match" => bool_score(norm_output == norm_ref),
            "contains_ref" => bool_score(norm_output.contains(&norm_ref) && !norm_ref.is_empty()),
            "jaccard" => jaccard_score(&norm_output, &norm_ref),
            "bleu" => bleu_n(&norm_output, &norm_ref, 1),
            "bleu4" => bleu_n(&norm_output, &norm_ref, 4),
            "rouge_l" => rouge_l_f1(&norm_output, &norm_ref),
            "rouge_lsum" => rouge_lsum_f1(&norm_output, &norm_ref),
            "f1" => token_f1(&norm_output, &norm_ref),
            _ => 0.0,
        };
        scores.insert(name.clone(), score);
    }

    scores
}

fn normalize(text: &str) -> String {
    text.split_whitespace()
        .map(|t| t.to_lowercase())
        .collect::<Vec<String>>()
        .join(" ")
        .trim()
        .to_string()
}

fn bool_score(value: bool) -> f64 {
    if value { 1.0 } else { 0.0 }
}

fn jaccard_score(output: &str, reference: &str) -> f64 {
    let out_tokens = tokenize_set(output);
    let ref_tokens = tokenize_set(reference);
    if out_tokens.is_empty() && ref_tokens.is_empty() {
        return 1.0;
    }
    if out_tokens.is_empty() || ref_tokens.is_empty() {
        return 0.0;
    }
    let intersection = out_tokens.intersection(&ref_tokens).count() as f64;
    let union = out_tokens.union(&ref_tokens).count() as f64;
    if union == 0.0 { 0.0 } else { intersection / union }
}

fn bleu_n(output: &str, reference: &str, max_n: usize) -> f64 {
    let out_tokens = tokenize_vec(output);
    let ref_tokens = tokenize_vec(reference);
    if out_tokens.is_empty() || ref_tokens.is_empty() {
        return 0.0;
    }

    let mut precisions: Vec<f64> = Vec::new();
    for n in 1..=max_n {
        let p = ngram_precision(&out_tokens, &ref_tokens, n);
        precisions.push(p.max(1e-9));
    }

    let log_sum: f64 = precisions.iter().map(|p| p.ln()).sum();
    let geo_mean = (log_sum / max_n as f64).exp();

    let bp = if out_tokens.len() > ref_tokens.len() {
        1.0
    } else {
        (1.0 - (ref_tokens.len() as f64 / out_tokens.len() as f64)).exp()
    };

    geo_mean * bp
}

fn ngram_precision(output: &[String], reference: &[String], n: usize) -> f64 {
    if output.len() < n {
        return 0.0;
    }
    let out_counts = ngram_counts(output, n);
    let ref_counts = ngram_counts(reference, n);

    let mut overlap = 0usize;
    let mut total = 0usize;
    for (gram, count) in out_counts {
        total += count;
        if let Some(ref_count) = ref_counts.get(&gram) {
            overlap += count.min(*ref_count);
        }
    }

    if total == 0 { 0.0 } else { overlap as f64 / total as f64 }
}

fn rouge_l_f1(output: &str, reference: &str) -> f64 {
    let out_tokens = tokenize_vec(output);
    let ref_tokens = tokenize_vec(reference);
    if out_tokens.is_empty() || ref_tokens.is_empty() {
        return 0.0;
    }

    let lcs = lcs_length(&out_tokens, &ref_tokens) as f64;
    let precision = lcs / out_tokens.len() as f64;
    let recall = lcs / ref_tokens.len() as f64;
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn rouge_lsum_f1(output: &str, reference: &str) -> f64 {
    let candidate_tokens = tokenize_vec(output);
    let reference_sentences = split_sentences(reference);
    if candidate_tokens.is_empty() || reference_sentences.is_empty() {
        return 0.0;
    }

    let mut matched = vec![false; candidate_tokens.len()];
    let mut total_ref_tokens = 0usize;

    for sentence in reference_sentences {
        let ref_tokens = tokenize_vec(&sentence);
        if ref_tokens.is_empty() {
            continue;
        }
        total_ref_tokens += ref_tokens.len();
        let indices = lcs_match_indices(&candidate_tokens, &ref_tokens);
        for idx in indices {
            if idx < matched.len() {
                matched[idx] = true;
            }
        }
    }

    if total_ref_tokens == 0 {
        return 0.0;
    }

    let union_lcs = matched.iter().filter(|v| **v).count() as f64;
    let precision = union_lcs / candidate_tokens.len() as f64;
    let recall = union_lcs / total_ref_tokens as f64;
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn token_f1(output: &str, reference: &str) -> f64 {
    let out_tokens = tokenize_vec(output);
    let ref_tokens = tokenize_vec(reference);
    if out_tokens.is_empty() || ref_tokens.is_empty() {
        return 0.0;
    }

    let out_counts = token_counts(&out_tokens);
    let ref_counts = token_counts(&ref_tokens);

    let mut overlap = 0usize;
    for (token, count) in out_counts.iter() {
        if let Some(ref_count) = ref_counts.get(token) {
            overlap += count.min(ref_count);
        }
    }

    let precision = overlap as f64 / out_tokens.len() as f64;
    let recall = overlap as f64 / ref_tokens.len() as f64;
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn tokenize_set(text: &str) -> HashSet<String> {
    text.split_whitespace()
        .map(|t| t.to_lowercase())
        .collect::<HashSet<String>>()
}

fn tokenize_vec(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|t| t.to_lowercase())
        .collect::<Vec<String>>()
}

fn token_counts(tokens: &[String]) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for token in tokens {
        *counts.entry(token.clone()).or_insert(0) += 1;
    }
    counts
}

fn ngram_counts(tokens: &[String], n: usize) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    if tokens.len() < n {
        return counts;
    }
    for i in 0..=tokens.len() - n {
        let gram = tokens[i..i + n].join(" ");
        *counts.entry(gram).or_insert(0) += 1;
    }
    counts
}

fn lcs_length(a: &[String], b: &[String]) -> usize {
    let mut dp = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            if a[i - 1] == b[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }
    dp[a.len()][b.len()]
}

fn lcs_match_indices(a: &[String], b: &[String]) -> Vec<usize> {
    let mut dp = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            if a[i - 1] == b[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    let mut i = a.len();
    let mut j = b.len();
    let mut indices: Vec<usize> = Vec::new();
    while i > 0 && j > 0 {
        if a[i - 1] == b[j - 1] {
            indices.push(i - 1);
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] >= dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }

    indices.reverse();
    indices
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if ch == '.' || ch == '!' || ch == '?' {
            if !current.trim().is_empty() {
                sentences.push(current.trim().to_string());
            }
            current.clear();
        }
    }
    if !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }
    sentences
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_and_f1() {
        let scores = compute_metrics(
            &vec!["exact_match".into(), "f1".into()],
            "hello world",
            "hello world",
        );
        assert_eq!(scores["exact_match"], 1.0);
        assert!((scores["f1"] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bleu4_penalizes_short_output() {
        let scores = compute_metrics(&vec!["bleu4".into()], "short", "short sentence here");
        assert!(scores["bleu4"] < 0.5);
    }

    #[test]
    fn rouge_lsum_handles_multi_sentence() {
        let scores = compute_metrics(
            &vec!["rouge_lsum".into()],
            "cats sit. dogs run.",
            "cats sit. dogs run.",
        );
        assert!(scores["rouge_lsum"] > 0.9);
    }
}
