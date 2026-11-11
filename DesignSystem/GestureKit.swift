import SwiftUI
import TesseraCore

/// The SwiftUI side of the gesture policy (docs/design/gesture-policy.md). Every swipe, page turn and pull in the app
/// goes through these three modifiers, so a touch is judged by `GestureTuning` and nothing else: the axis is locked
/// once, an action never starts in the back-edge zone, a flick needs real speed, and the arm tick is the house tick.

// MARK: Row swipe

/// What a row swipe tells its row.
enum RowSwipeEvent: Equatable {
    /// The pull to draw changed (signed: right is positive).
    case pulled(CGFloat)
    /// The pull crossed the arm distance; the tick has played.
    case armed
    /// The pull fell back under the arm distance.
    case disarmed
    /// Let go armed (or on a real flick): do the action. `.ended` follows.
    case committed(SwipeSides)
    /// Let go short of it, taken back, or cancelled by the system: the row returns. `.ended` follows.
    case cancelled
    /// Always the last event of a touch, after `committed` or `cancelled`.
    case ended
}

private struct RowSwipe: ViewModifier {
    let sides: SwipeSides
    let arm: CGFloat
    let limit: CGFloat
    let allowsFlick: Bool
    let onEvent: (RowSwipeEvent) -> Void

    @State private var tracker = SwipeTracker()
    @State private var cue = HapticCue()
    @GestureState private var touching = false

    func body(content: Content) -> some View {
        content
            .simultaneousGesture(
                DragGesture(minimumDistance: CGFloat(GestureTuning.dragStart), coordinateSpace: .global)
                    .updating($touching) { _, state, _ in state = true }
                    .onChanged(changed)
                    .onEnded(ended)
            )
            // A touch the system took (a scroll won) never reaches `onEnded`; the row still has to come home.
            .onChange(of: touching) { _, isTouching in
                guard !isTouching, tracker.lock != .open else { return }
                tracker.reset()
                onEvent(.cancelled)
                onEvent(.ended)
            }
            .tesseraFeedback(cue)
    }

    private func changed(_ value: DragGesture.Value) {
        guard let pull = tracker.update(
            startX: Double(value.startLocation.x), dx: Double(value.translation.width),
            dy: Double(value.translation.height), sides: sides, limit: Double(limit)
        ) else { return }
        let wasArmed = tracker.armed
        let justArmed = tracker.arm(at: Double(arm))
        onEvent(.pulled(CGFloat(pull)))
        if justArmed {
            cue.play(.selection)
            onEvent(.armed)
        } else if wasArmed, !tracker.armed {
            onEvent(.disarmed)
        }
    }

    private func ended(_ value: DragGesture.Value) {
        let outcome = tracker.outcome(velocityX: Double(value.velocity.width), arm: Double(arm),
                                      allowsFlick: allowsFlick)
        tracker.reset()
        switch outcome {
        case let .commit(side): onEvent(.committed(side))
        case .cancel: onEvent(.cancelled)
        }
        onEvent(.ended)
    }
}

// MARK: Page swipe

private struct PageSwipe: ViewModifier {
    let onTurn: (_ next: Bool) -> Void

    func body(content: Content) -> some View {
        content.simultaneousGesture(
            DragGesture(minimumDistance: CGFloat(GestureTuning.dragStart), coordinateSpace: .global)
                .onEnded { value in
                    let turn = GestureTuning.pageTurn(
                        startX: Double(value.startLocation.x), dx: Double(value.translation.width),
                        dy: Double(value.translation.height), velocityX: Double(value.velocity.width)
                    )
                    if turn != 0 {
                        onTurn(turn > 0)
                    }
                }
        )
    }
}

// MARK: Pull

private struct PullGesture: ViewModifier {
    let onPull: (_ down: Bool) -> Void

    func body(content: Content) -> some View {
        content.simultaneousGesture(
            DragGesture(minimumDistance: CGFloat(GestureTuning.dragStart), coordinateSpace: .global)
                .onEnded { value in
                    let pull = GestureTuning.pull(
                        dx: Double(value.translation.width), dy: Double(value.translation.height),
                        velocityY: Double(value.velocity.height)
                    )
                    if pull != 0 {
                        onPull(pull > 0)
                    }
                }
        )
    }
}

extension View {
    /// A row's swipe. `arm` is the pull at which the tick plays and a release commits; `limit` holds the pull. The row
    /// draws `.pulled`, acts on `.committed` and comes home on `.cancelled`; `.ended` is its cue to clear flags such as
    /// "a swipe just happened, so the tap that follows is not a tap". A row whose action has no Undo passes
    /// `allowsFlick: false`, so only a full pull commits it.
    func rowSwipe(
        _ sides: SwipeSides,
        arm: CGFloat = CGFloat(GestureTuning.armDistance),
        limit: CGFloat = CGFloat(GestureTuning.swipeLimit),
        allowsFlick: Bool = true,
        onEvent: @escaping (RowSwipeEvent) -> Void
    ) -> some View {
        modifier(RowSwipe(sides: sides, arm: arm, limit: limit, allowsFlick: allowsFlick, onEvent: onEvent))
    }

    /// A side swipe that turns a page: `next` is true for a drag to the left. Runs alongside the page's own scroll.
    func pageSwipe(_ onTurn: @escaping (_ next: Bool) -> Void) -> some View {
        modifier(PageSwipe(onTurn: onTurn))
    }

    /// A pull down (`down` true) or up that opens or closes something. Runs alongside the page's own scroll.
    func pullGesture(_ onPull: @escaping (_ down: Bool) -> Void) -> some View {
        modifier(PullGesture(onPull: onPull))
    }
}
