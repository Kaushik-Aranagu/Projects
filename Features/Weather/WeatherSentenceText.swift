import SwiftUI

// CLOUD-WRITTEN, NOT COMPILED. How the room sets a sentence. `WeatherSentenceText` draws the landing's two lines
// (weather words in their colors; Builder A's weather-word effects replace the plain color here and nowhere else).
// `WeatherDoorSentence` is "The week": every weather word is a door to its day with a 44 pt hit area.

struct WeatherSentenceText: View {
    let sentence: WeatherSentence
    let size: CGFloat
    var weight: Font.Weight = .medium
    var ink: Color = Theme.Colors.ink

    var body: some View {
        Text(attributed)
            .font(WeatherType.figtree(size, weight, relativeTo: .title3))
            .fixedSize(horizontal: false, vertical: true)
            .accessibilityLabel(sentence.text)
    }

    private var attributed: AttributedString {
        var result = AttributedString()
        for piece in sentence.pieces {
            switch piece {
            case let .plain(text):
                var part = AttributedString(text)
                part.foregroundColor = ink
                result += part
            case let .word(kind, text, _):
                var part = AttributedString(text)
                part.foregroundColor = kind.color
                result += part
            }
        }
        return result
    }
}

// MARK: - The week sentence

struct WeatherDoorSentence: View {
    let sentence: WeatherSentence
    let size: CGFloat
    let lineHeight: CGFloat
    let open: (Int) -> Void

    private struct Token: Identifiable {
        let id: Int
        let text: String
        let kind: WeatherKind?
        let day: Int?
        let glued: Bool
    }

    private var tokens: [Token] {
        var out: [Token] = []
        var previousEndsWithSpace = true
        for piece in sentence.pieces {
            switch piece {
            case let .plain(text):
                let parts = text.split(separator: " ", omittingEmptySubsequences: true)
                for (position, part) in parts.enumerated() {
                    let glued = position == 0 && !text.hasPrefix(" ") && !previousEndsWithSpace
                    out.append(Token(id: out.count, text: String(part), kind: nil, day: nil, glued: glued))
                }
                if !text.isEmpty { previousEndsWithSpace = text.hasSuffix(" ") }
            case let .word(kind, text, day):
                out.append(Token(id: out.count, text: text, kind: kind, day: day, glued: !previousEndsWithSpace))
                previousEndsWithSpace = false
            }
        }
        return out
    }

    var body: some View {
        WeatherFlow(lineHeight: lineHeight, space: size * 0.27) {
            ForEach(tokens) { token in
                if let day = token.day, let kind = token.kind {
                    Button { open(day) } label: {
                        Text(token.text)
                            .font(WeatherType.figtree(size, .medium, relativeTo: .title3))
                            .foregroundStyle(kind.color)
                            .frame(minWidth: 44, minHeight: 44)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .layoutValue(key: WeatherGlue.self, value: token.glued)
                    .accessibilityLabel("\(token.text), open the day")
                    .accessibilityIdentifier("weather.door.\(day)")
                } else {
                    Text(token.text)
                        .font(WeatherType.figtree(size, .medium, relativeTo: .title3))
                        .foregroundStyle(token.kind?.color ?? Theme.Colors.ink)
                        .layoutValue(key: WeatherGlue.self, value: token.glued)
                        .accessibilityHidden(true)
                }
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(sentence.text)
    }
}

private struct WeatherGlue: LayoutValueKey {
    static let defaultValue = false
}

/// Words flowing left to right, wrapping at the width. Each line is `lineHeight` tall whatever its words measure, so
/// a door's 44 pt hit area overlaps its neighbours' lines instead of spreading them apart.
private struct WeatherFlow: Layout {
    let lineHeight: CGFloat
    let space: CGFloat

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache _: inout ()) -> CGSize {
        let width = proposal.width ?? 340
        let lines = place(in: width, subviews: subviews).lines
        return CGSize(width: width, height: CGFloat(lines) * lineHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal _: ProposedViewSize, subviews: Subviews, cache _: inout ()) {
        let placed = place(in: bounds.width, subviews: subviews)
        for (index, origin) in placed.origins.enumerated() {
            subviews[index].place(
                at: CGPoint(x: bounds.minX + origin.x, y: bounds.minY + origin.y + lineHeight / 2),
                anchor: .leading, proposal: .unspecified
            )
        }
    }

    private func place(in width: CGFloat, subviews: Subviews) -> (origins: [CGPoint], lines: Int) {
        var origins: [CGPoint] = []
        var x: CGFloat = 0
        var line = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            let gap = subview[WeatherGlue.self] || origins.isEmpty ? 0 : space
            if x > 0, x + gap + size.width > width {
                line += 1
                x = 0
                origins.append(CGPoint(x: 0, y: CGFloat(line) * lineHeight))
                x = size.width
            } else {
                origins.append(CGPoint(x: x + (x > 0 ? gap : 0), y: CGFloat(line) * lineHeight))
                x += (x > 0 ? gap : 0) + size.width
            }
        }
        return (origins, origins.isEmpty ? 0 : line + 1)
    }
}
