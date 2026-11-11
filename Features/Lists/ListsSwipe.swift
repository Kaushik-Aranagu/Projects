import SwiftUI
import TesseraCore

/// A list row's left swipe (lists-acceptance 13.6): REMOVE rises in the live colour from 60 pt and commits past 96.
/// Never a hold, a red row, a confirm or a trash can. VoiceOver gets the same thing as the Remove action.
struct SwipeToRemove: ViewModifier {
    /// Removes the row; true when it went. The list page plays the haptic, since the row is gone by then (QA on #786).
    let onRemove: () -> Bool
    /// The word that rises: "Remove" on a list, "Forget" on What Tess knows.
    var word = "Remove"
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var pull: CGFloat = 0
    /// The swipe removed the row: it keeps its pull while it leaves, so it never flashes back first.
    @State private var wentAway = false

    static let rises: CGFloat = 60
    static let commits: CGFloat = 96

    /// How much of REMOVE shows at a pull: nothing before 60 pt, all of it at 96.
    static func reveal(_ pull: CGFloat) -> Double {
        Double(min(max((abs(pull) - rises) / (commits - rises), 0), 1))
    }

    func body(content: Content) -> some View {
        content
            .accessibilityAction(named: Text(word)) { _ = remove() }
            .offset(x: pull)
            .background(alignment: .trailing) {
                Text(word)
                    .textRole(.labelStrong)
                    .textCase(.uppercase)
                    .foregroundStyle(Theme.Colors.live)
                    .opacity(Self.reveal(pull))
                    .offset(y: reduceMotion ? 0 : 6 * (1 - Self.reveal(pull)))
                    .accessibilityHidden(true)
            }
            .rowSwipe(.trailing, arm: Self.commits, limit: Self.commits + 24) { event in
                switch event {
                case let .pulled(value): pull = value
                case .committed: wentAway = remove()
                case .ended:
                    if !wentAway, pull != 0 {
                        withTesseraAnimation(reduceMotion: reduceMotion) { pull = 0 }
                    }
                    wentAway = false
                case .cancelled, .armed, .disarmed: break
                }
            }
    }

    private func remove() -> Bool {
        onRemove()
    }
}

extension View {
    func swipeToRemove(word: String = "Remove", _ onRemove: @escaping () -> Bool) -> some View {
        modifier(SwipeToRemove(onRemove: onRemove, word: word))
    }
}
