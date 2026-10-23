import Foundation

/// Supported languages for script-based filtering.
public enum Language: String, Sendable, CaseIterable {
    case english = "en"
    case spanish = "es"
    case french = "fr"
    case german = "de"
    case italian = "it"
    case portuguese = "pt"
    case romanian = "ro"  // uses ș, ț (Latin Extended-B)
    case dutch = "nl"
    case danish = "da"
    case swedish = "sv"
    case finnish = "fi"
    case hungarian = "hu"
    case estonian = "et"
    case latvian = "lv"
    case lithuanian = "lt"
    case maltese = "mt"

    // Latin-script Slavic — prone to Cyrillic confusion in multilingual ASR.
    case polish = "pl"
    case czech = "cs"
    case slovak = "sk"
    case slovenian = "sl"
    case croatian = "hr"
    case bosnian = "bs"

    case russian = "ru"
    case ukrainian = "uk"
    case belarusian = "be"
    case bulgarian = "bg"
    case serbian = "sr"

    // Greek script
    case greek = "el"

    public var script: Script {
        switch self {
        case .english, .spanish, .french, .german, .italian, .portuguese, .romanian,
            .dutch, .danish, .swedish, .finnish, .hungarian, .estonian, .latvian,
            .lithuanian, .maltese,
            .polish, .czech, .slovak, .slovenian, .croatian, .bosnian:
            return .latin
        case .russian, .ukrainian, .belarusian, .bulgarian, .serbian:
            return .cyrillic
        case .greek:
            return .greek
        }
    }

    /// The letters beyond ASCII this language writes, lowercase; `nil` where
    /// every letter of its script is its own. Every Latin-script language also
    /// writes ASCII letters (loanwords, names, English), so Polish keeps
    /// English words and English keeps no others.
    var letters: Set<Character>? {
        switch self {
        case .english: return []
        case .spanish: return Set("áéíóúüñ")
        case .french: return Set("àâæçéèêëîïôœùûüÿ")
        case .german: return Set("äöüß")
        case .italian: return Set("àèéìíîòóùú")
        case .portuguese: return Set("áâãàçéêíóôõúü")
        case .romanian: return Set("ăâîșțşţ")
        case .dutch: return Set("áéíóúàèëïöü")
        case .danish: return Set("æøåé")
        case .swedish: return Set("åäöé")
        case .finnish: return Set("åäöšž")
        case .hungarian: return Set("áéíóöőúüű")
        case .estonian: return Set("äöõüšž")
        case .latvian: return Set("āčēģīķļņšūž")
        case .lithuanian: return Set("ąčęėįšųūž")
        case .maltese: return Set("àċèġħìòùż")
        case .polish: return Set("ąćęłńóśźż")
        case .czech: return Set("áčďéěíňóřšťúůýž")
        case .slovak: return Set("áäčďéíĺľňóôŕšťúýž")
        case .slovenian: return Set("čšž")
        case .croatian, .bosnian: return Set("čćđšž")
        case .russian, .ukrainian, .belarusian, .bulgarian, .serbian, .greek: return nil
        }
    }
}

/// Writing script categories. Latin and Cyrillic letter ranges are
/// non-overlapping, which the `matches` impl relies on.
public enum Script: Sendable {
    case latin
    case cyrillic
    case greek
}

/// Filters ASR decoder tokens against the target languages' alphabets.
///
/// Used by the v3 TDT decoder to stop the joint network from emitting
/// wrong-language tokens: Cyrillic letters while transcribing Polish
/// (issue #512), and within the Latin script, Czech `ř` or Hungarian `ő`
/// where the audio is unclear and the model drifts into a language it
/// was never asked for.
internal struct TokenLanguageFilter: Sendable {

    /// SentencePiece word-boundary marker (▁, U+2581). Stripped before script
    /// checks — carries no script information.
    private static let sentencePieceBoundary: Unicode.Scalar = "\u{2581}"

    /// Whether every character in `text` is compatible with `script`.
    ///
    /// The Latin path has no "reject Cyrillic" guard because the Cyrillic
    /// block sits outside every Latin range. The Cyrillic path *does* need an
    /// explicit ASCII-letter reject — ASCII overlaps with the Latin range, so
    /// without the guard a token like `"cat"` would pass as Cyrillic.
    static func matches(_ text: String, script: Script) -> Bool {
        let cleanedText = text.replacingOccurrences(of: String(sentencePieceBoundary), with: "")

        // Pure boundary markers are script-neutral — let the caller's argmax
        // decide on logit alone rather than excluding them here.
        guard !cleanedText.isEmpty else { return true }

        let chars = cleanedText.unicodeScalars
        switch script {
        case .latin:
            return chars.allSatisfy {
                ($0.value >= 0x0020 && $0.value <= 0x007F)  // ASCII
                    || ($0.value >= 0x00A0 && $0.value <= 0x00FF)  // Latin-1
                    || ($0.value >= 0x0100 && $0.value <= 0x017F)  // Latin Extended-A
                    || ($0.value >= 0x0180 && $0.value <= 0x024F)  // Latin Extended-B
                    || ($0.value >= 0x0300 && $0.value <= 0x036F)  // Combining Diacritical Marks (NFD)
                    || ($0.value >= 0x1E00 && $0.value <= 0x1EFF)  // Latin Extended Additional
            }
        case .cyrillic:
            return chars.allSatisfy { char in
                let value = char.value
                if value >= 0x0400 && value <= 0x04FF { return true }
                // ASCII is script-neutral except for letters A-Z/a-z, which
                // must be rejected (see function doc).
                if value >= 0x0020 && value <= 0x007F {
                    if (value >= 0x41 && value <= 0x5A) || (value >= 0x61 && value <= 0x7A) {
                        return false
                    }
                    return true
                }
                return false
            }
        case .greek:
            return chars.allSatisfy { char in
                let value = char.value
                // Greek and Coptic block (U+0370–U+03FF); Coptic subset not used
                if value >= 0x0370 && value <= 0x03FF { return true }
                // Greek Extended (polytonic/ancient Greek diacritics)
                if value >= 0x1F00 && value <= 0x1FFF { return true }
                // Combining Diacritical Marks (NFD)
                if value >= 0x0300 && value <= 0x036F { return true }
                // Allow ASCII punctuation, digits, spaces (script-neutral)
                if value >= 0x0020 && value <= 0x007F {
                    // Reject Latin letters A-Z/a-z
                    if (value >= 0x41 && value <= 0x5A) || (value >= 0x61 && value <= 0x7A) {
                        return false
                    }
                    return true
                }
                return false
            }

        }
    }

    /// Whether `text` is written in the letters of any of `languages`: its
    /// script, and for a Latin-script language, ASCII and that language's own
    /// letters.
    static func matches(_ text: String, languages: Set<Language>) -> Bool {
        languages.contains { matches(text, language: $0) }
    }

    static func matches(_ text: String, language: Language) -> Bool {
        guard matches(text, script: language.script) else { return false }
        guard let letters = language.letters else { return true }
        return text.lowercased().allSatisfy { !$0.isLetter || $0.isASCII || letters.contains($0) }
    }

    /// Filter top-K candidates by language and return the highest-logit match.
    ///
    /// Returned probability is softmax **over the top-K logits only**, not the
    /// full vocabulary. It is systematically larger than a full-vocab softmax
    /// — use it for relative ranking, not as a drop-in probability.
    ///
    /// - Returns: `nil` if no right-language match exists or inputs are empty.
    static func filterTopK(
        topKIds: [Int],
        topKLogits: [Float],
        vocabulary: [Int: String],
        languages: Set<Language>
    ) -> (tokenId: Int, probability: Float)? {
        let count = min(topKIds.count, topKLogits.count)
        guard count > 0 else { return nil }

        // Explicit argmax over right-language candidates. CoreML top-K is not
        // guaranteed sorted. The `bestIdx < 0` clause forces the first match
        // to win even when its logit is -infinity (otherwise it would never
        // beat the sentinel).
        var bestIdx: Int = -1
        var bestLogit: Float = -.infinity
        for idx in 0..<count {
            let tokenId = topKIds[idx]
            guard let tokenText = vocabulary[tokenId] else { continue }
            guard matches(tokenText, languages: languages) else { continue }

            let logit = topKLogits[idx]
            if bestIdx < 0 || logit > bestLogit {
                bestLogit = logit
                bestIdx = idx
            }
        }
        guard bestIdx >= 0 else { return nil }

        // Top-K softmax with max-logit stability.
        var maxLogit = -Float.infinity
        for i in 0..<count where topKLogits[i] > maxLogit {
            maxLogit = topKLogits[i]
        }
        guard maxLogit.isFinite else { return (topKIds[bestIdx], 0) }

        var sumExp: Float = 0
        for i in 0..<count {
            sumExp += expf(topKLogits[i] - maxLogit)
        }
        guard sumExp > 0 else { return (topKIds[bestIdx], 0) }

        let prob = expf(topKLogits[bestIdx] - maxLogit) / sumExp
        return (topKIds[bestIdx], max(0, min(1, prob)))
    }
}
