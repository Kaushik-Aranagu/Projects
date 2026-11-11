import Foundation

/// The app's one gesture policy (docs/design/gesture-policy.md). Every number a touch is judged by lives here, so a
/// swipe, a tap and a hold feel the same in every room, and the feel is pinned by tests. Pure Doubles in points,
/// seconds and points per second; `DesignSystem/GestureKit.swift` is the SwiftUI side that reads them.
public enum GestureTuning {
    // MARK: Distances

    /// A drag that starts an action does not begin before this much travel (SwiftUI `minimumDistance`).
    public static let dragStart = 10.0
    /// The axis of a swipe is decided once, at this much travel, and holds until the finger lifts.
    public static let axisLock = 12.0
    /// A sideways swipe must lead the vertical drift by this ratio to lock horizontal. Anything closer is a scroll.
    public static let horizontalBias = 1.5
    /// A touch that moves less than this is a tap, not a drag.
    public static let tapSlop = 10.0
    /// A hold is cancelled by movement past this.
    public static let holdSlop = 8.0

    // MARK: Commits

    /// A row swipe arms past this pull (the house tick plays once) and commits on release while armed.
    public static let armDistance = 96.0
    /// A swipe whose row does not follow the finger (nothing is drawn, only the tick tells) arms a little earlier.
    public static let armUndrawn = 72.0
    /// Once armed, the pull must fall this far back before it disarms, so the tick never chatters at the line.
    public static let disarmHysteresis = 8.0
    /// How far a row follows the finger at most.
    public static let swipeLimit = 144.0
    /// A page turn (a side swipe between pages) commits past this.
    public static let pageDistance = 60.0
    /// A pull down (or up) that opens or closes a sheet, or a header's order, commits past this.
    public static let pullDistance = 56.0
    /// A page swipe or a pull must lead the other axis by this ratio.
    public static let pagerBias = 2.0
    /// A flick commits short of the full distance only at this speed in the swipe's own direction ...
    public static let flickVelocity = 500.0
    /// ... and only after at least this much travel, so a twitch never commits.
    public static let flickTravel = 28.0
    /// A release that is moving back against the pull at this speed cancels, even past the arm distance.
    public static let reverseVelocity = 350.0

    // MARK: Edges

    /// The system back gesture owns the leading edge out to here.
    public static let backEdge = 24.0
    /// An action that starts a rightward drag (an action word, a Done swipe) must begin at least this far in.
    public static let actionInset = 28.0
    /// The home indicator and Control Center own the bottom and top of the screen; a drag that is an action starts
    /// at least this far in from them.
    public static let verticalEdgeInset = 24.0

    // MARK: Action words and stacks

    /// An action word (the bottom-left word of a room) leans toward an option after this much drag ...
    public static let actionLean = 18.0
    /// ... and rolls to the option's name past this; letting go while rolled commits, with Undo.
    public static let actionRoll = 56.0
    /// A stack of widgets or things commits a swipe past this share of its width.
    public static let stackCommitFraction = 0.35

    public enum ActionWordStage: Sendable, Equatable {
        case rest, lean, rolled
    }

    /// Where an action word is for a drag of `travel` points toward one of its options (negative or zero is rest).
    public static func actionWordStage(travel: Double) -> ActionWordStage {
        if travel >= actionRoll {
            return .rolled
        }
        return travel >= actionLean ? .lean : .rest
    }

    /// Whether letting go commits the option: it is rolled and the finger is not going back.
    public static func actionWordCommits(travel: Double, velocity: Double) -> Bool {
        actionWordStage(travel: travel) == .rolled && velocity > -reverseVelocity
    }

    /// Whether a stack swipe of `travel` points across `width` commits: past the share, or a real flick.
    public static func stackCommits(travel: Double, width: Double, velocity: Double) -> Bool {
        guard width > 0 else { return false }
        return commits(distance: travel, velocity: velocity, arm: width * stackCommitFraction)
    }

    // MARK: Time

    /// A tap right after a custom scroller (a wheel, a pager) lets go is the finger stopping it, not a tap.
    public static let settleHoldoff = 0.12
    /// Hold tiers. Quick picks something up; pick starts a drag or a reorder; open opens the detail.
    public static let holdQuick = 0.18
    public static let holdPick = 0.35
    public static let holdOpen = 0.5
    /// Every commit-on-release offers Undo for this long (the op-log Undo, 10 s everywhere).
    public static let undoSeconds = 10.0

    // MARK: Axis

    public enum Axis: Sendable, Equatable {
        case horizontal, vertical
    }

    /// Nil until `axisLock` of travel; then horizontal only if it leads by `bias`, else vertical.
    public static func axis(dx: Double, dy: Double, bias: Double = horizontalBias) -> Axis? {
        guard hypot(dx, dy) >= axisLock else { return nil }
        return abs(dx) >= abs(dy) * bias ? .horizontal : .vertical
    }

    /// Whether a drag that began at `x` may be an action that moves right: it must clear the back-edge gesture.
    public static func clearsBackEdge(startX x: Double) -> Bool {
        x >= actionInset
    }

    /// Whether a touch is a tap: it stayed within the slop.
    public static func isTap(dx: Double, dy: Double) -> Bool {
        hypot(dx, dy) <= tapSlop
    }

    /// A side swipe that turns a page: +1 for the next page (a drag to the left), -1 for the previous, 0 for none. It
    /// must lock horizontal with `pagerBias`, go past `pageDistance` (or flick), and a drag to the right must start
    /// clear of the back edge. `velocityX` is the speed at lift, positive to the right.
    public static func pageTurn(startX: Double, dx: Double, dy: Double, velocityX: Double) -> Int {
        guard axis(dx: dx, dy: dy, bias: pagerBias) == .horizontal else { return 0 }
        if dx > 0, !clearsBackEdge(startX: startX) {
            return 0
        }
        let direction = dx < 0 ? -1.0 : 1.0
        guard commits(distance: abs(dx), velocity: velocityX * direction, arm: pageDistance) else { return 0 }
        return dx < 0 ? 1 : -1
    }

    /// A vertical pull: +1 for a pull down, -1 for a pull up, 0 for none. It must lock vertical with `pagerBias` and
    /// go past `pullDistance` (or flick). `velocityY` is the speed at lift, positive downward.
    public static func pull(dx: Double, dy: Double, velocityY: Double) -> Int {
        guard hypot(dx, dy) >= axisLock, abs(dy) >= abs(dx) * pagerBias else { return 0 }
        let sign = dy > 0 ? 1.0 : -1.0
        guard commits(distance: abs(dy), velocity: velocityY * sign, arm: pullDistance) else { return 0 }
        return dy > 0 ? 1 : -1
    }

    /// Whether a release commits. `distance` is the signed travel along the swipe's axis and `velocity` the signed
    /// speed at lift, both in the commit direction's sign (positive = toward the commit). Past `arm` commits unless
    /// the finger is going back; short of `arm` commits only as a clear flick.
    public static func commits(distance: Double, velocity: Double, arm: Double) -> Bool {
        if distance >= arm {
            return velocity > -reverseVelocity
        }
        return distance >= flickTravel && velocity >= flickVelocity
    }
}

/// Which way a row may be swiped. `leading` is a drag to the right (Done), `trailing` a drag to the left (Not now,
/// Remove, To get).
public struct SwipeSides: OptionSet, Sendable {
    public let rawValue: Int
    public init(rawValue: Int) {
        self.rawValue = rawValue
    }

    public static let leading = SwipeSides(rawValue: 1)
    public static let trailing = SwipeSides(rawValue: 2)
    public static let both: SwipeSides = [.leading, .trailing]
}

public enum SwipeOutcome: Sendable, Equatable {
    case cancel
    case commit(SwipeSides)
}

/// One row swipe, from touch-down to lift. It owns the three things every swipe used to decide for itself: the axis
/// (locked once, never flipped by later drift), the arm (with hysteresis, so the tick fires once), and the commit
/// (distance or a real flick, never a reverse). The view feeds it the drag and draws `pull`.
public struct SwipeTracker: Sendable, Equatable {
    public enum Lock: Sendable, Equatable {
        case open, horizontal, rejected
    }

    public private(set) var lock = Lock.open
    public private(set) var armed = false
    /// The signed pull to draw: 0 while undecided or rejected, then the travel since the lock, held inside `limit`.
    public private(set) var pull = 0.0
    private var origin = 0.0

    public init() {}

    /// Feeds one drag sample. Returns the pull to draw, or nil when this drag is not a swipe (a scroll, a drag that
    /// started in the back-edge zone, or a swipe toward a side the row does not allow), so the view leaves it alone.
    @discardableResult
    public mutating func update(startX: Double, dx: Double, dy: Double, sides: SwipeSides, limit: Double) -> Double? {
        if lock == .open {
            guard let axis = GestureTuning.axis(dx: dx, dy: dy) else { return nil }
            let blocked = dx > 0 && !GestureTuning.clearsBackEdge(startX: startX)
            lock = axis == .horizontal && !blocked ? .horizontal : .rejected
            origin = dx
        }
        guard lock == .horizontal else { return nil }
        let travel = dx - origin
        let allowed = travel > 0 ? sides.contains(.leading) : sides.contains(.trailing)
        pull = allowed ? min(max(travel, -limit), limit) : 0
        return pull
    }

    /// Updates `armed` for a pull; true when it just armed (play the tick once).
    @discardableResult
    public mutating func arm(at arm: Double = GestureTuning.armDistance) -> Bool {
        let magnitude = abs(pull)
        if armed {
            armed = magnitude >= arm - GestureTuning.disarmHysteresis
            return false
        }
        armed = magnitude >= arm
        return armed
    }

    /// The outcome at lift, given the horizontal velocity at lift (positive = to the right). A row that has no Undo
    /// passes `allowsFlick: false`: only a pull past the arm distance (and not being taken back) commits it.
    public func outcome(velocityX: Double, arm: Double = GestureTuning.armDistance,
                        allowsFlick: Bool = true) -> SwipeOutcome
    {
        guard lock == .horizontal, pull != 0 else { return .cancel }
        let sign = pull > 0 ? 1.0 : -1.0
        let side: SwipeSides = pull > 0 ? .leading : .trailing
        let go = allowsFlick
            ? GestureTuning.commits(distance: abs(pull), velocity: velocityX * sign, arm: arm)
            : abs(pull) >= arm && velocityX * sign > -GestureTuning.reverseVelocity
        return go ? .commit(side) : .cancel
    }

    public mutating func reset() {
        self = SwipeTracker()
    }
}

/// A short hold-off after a custom scroller lets go, so the tap that stops a wheel is not read as a tap on the row it
/// stopped over. The system `ScrollView` already does this for its own scrolling; this is for wheels and pagers.
public struct TapHoldoff: Sendable, Equatable {
    private var lastMotion: Double?

    public init() {}

    /// Call when a drag or a coast ends (and while it moves).
    public mutating func moved(at time: Double) {
        lastMotion = time
    }

    /// Whether a tap at `time` counts.
    public func allowsTap(at time: Double) -> Bool {
        guard let lastMotion else { return true }
        return time - lastMotion >= GestureTuning.settleHoldoff
    }
}
