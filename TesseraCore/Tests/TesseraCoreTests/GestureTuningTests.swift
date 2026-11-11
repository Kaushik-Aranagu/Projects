@testable import TesseraCore
import Testing

/// The gesture policy (docs/design/gesture-policy.md): the numbers every swipe, tap and hold is judged by.
struct GestureTuningTests {
    // MARK: Axis

    @Test func noAxisBeforeTwelvePointsOfTravel() {
        #expect(GestureTuning.axis(dx: 8, dy: 3) == nil)
        #expect(GestureTuning.axis(dx: 0, dy: 11.9) == nil)
    }

    @Test func aClearSidewaysDragLocksHorizontalAndADiagonalStaysAScroll() {
        #expect(GestureTuning.axis(dx: 20, dy: 4) == .horizontal)
        #expect(GestureTuning.axis(dx: -18, dy: 6) == .horizontal)
        // 20 across and 16 down is a diagonal: it does not lead by 1.5x, so it is a scroll.
        #expect(GestureTuning.axis(dx: 20, dy: 16) == .vertical)
        #expect(GestureTuning.axis(dx: 3, dy: 30) == .vertical)
    }

    @Test func aTapStaysInsideTheSlop() {
        #expect(GestureTuning.isTap(dx: 4, dy: 5))
        #expect(!GestureTuning.isTap(dx: 9, dy: 9))
    }

    @Test func theBackEdgeIsClearOfActionDrags() {
        #expect(!GestureTuning.clearsBackEdge(startX: 12))
        #expect(!GestureTuning.clearsBackEdge(startX: 27.9))
        #expect(GestureTuning.clearsBackEdge(startX: 28))
    }

    // MARK: Commits

    @Test func pastTheArmDistanceCommitsUnlessTheFingerIsGoingBack() {
        #expect(GestureTuning.commits(distance: 80, velocity: 0, arm: 72))
        #expect(GestureTuning.commits(distance: 80, velocity: -300, arm: 72))
        #expect(!GestureTuning.commits(distance: 80, velocity: -400, arm: 72))
    }

    @Test func shortOfTheArmDistanceOnlyARealFlickCommits() {
        #expect(!GestureTuning.commits(distance: 40, velocity: 300, arm: 72))
        #expect(GestureTuning.commits(distance: 40, velocity: 600, arm: 72))
        // A twitch: fast but under the flick travel.
        #expect(!GestureTuning.commits(distance: 14, velocity: 900, arm: 72))
        // A flick against the direction never commits.
        #expect(!GestureTuning.commits(distance: 40, velocity: -900, arm: 72))
    }

    // MARK: Tracker

    private func swipe(
        _ samples: [(dx: Double, dy: Double)],
        startX: Double = 120,
        sides: SwipeSides = .both,
        limit: Double = 150
    ) -> SwipeTracker {
        var tracker = SwipeTracker()
        for sample in samples {
            tracker.update(startX: startX, dx: sample.dx, dy: sample.dy, sides: sides, limit: limit)
            tracker.arm()
        }
        return tracker
    }

    @Test func aVerticalScrollNeverMovesTheRow() {
        var tracker = SwipeTracker()
        #expect(tracker.update(startX: 120, dx: 2, dy: 30, sides: .both, limit: 150) == nil)
        // Drifting sideways later does not turn the scroll into a swipe.
        #expect(tracker.update(startX: 120, dx: 90, dy: 60, sides: .both, limit: 150) == nil)
        #expect(tracker.lock == .rejected)
        #expect(tracker.pull == 0)
        #expect(tracker.outcome(velocityX: 0) == .cancel)
    }

    @Test func thePullStartsAtTheLockNotWithAJump() {
        var tracker = SwipeTracker()
        #expect(tracker.update(startX: 120, dx: 8, dy: 0, sides: .both, limit: 150) == nil)
        #expect(tracker.update(startX: 120, dx: 14, dy: 1, sides: .both, limit: 150) == 0)
        #expect(tracker.update(startX: 120, dx: 44, dy: 2, sides: .both, limit: 150) == 30)
    }

    @Test func aHorizontalLockHoldsWhenTheFingerDriftsDown() {
        var tracker = swipe([(14, 0), (60, 20), (110, 90)])
        #expect(tracker.lock == .horizontal)
        #expect(tracker.pull == 96)
        #expect(tracker.outcome(velocityX: 0) == .commit(.leading))
        tracker.reset()
        #expect(tracker.lock == .open)
    }

    @Test func aSwipeToASideTheRowDoesNotHaveGoesNowhere() {
        let tracker = swipe([(-14, 0), (-60, 0), (-120, 0)], sides: .leading)
        #expect(tracker.pull == 0)
        #expect(tracker.outcome(velocityX: -400) == .cancel)
    }

    @Test func aRightwardDragFromTheBackEdgeBelongsToTheSystem() {
        let tracker = swipe([(14, 0), (60, 0), (120, 0)], startX: 10, sides: .leading)
        #expect(tracker.lock == .rejected)
        #expect(tracker.outcome(velocityX: 800) == .cancel)
        // A leftward swipe from the same edge is fine.
        let left = swipe([(-14, 0), (-60, 0), (-120, 0)], startX: 10, sides: .trailing)
        #expect(left.outcome(velocityX: 0) == .commit(.trailing))
    }

    @Test func theArmTickFiresOnceAndDoesNotChatterAtTheLine() {
        var tracker = SwipeTracker()
        var ticks = 0
        for dx in [14.0, 60, 110, 104, 120, 98, 105, 125] {
            tracker.update(startX: 120, dx: dx, dy: 0, sides: .both, limit: 150)
            if tracker.arm() { ticks += 1 }
        }
        // Armed at a pull of 96 (one tick); it held through 90 and 106, disarmed at 84, stayed off at 91, armed at 111.
        #expect(ticks == 2)
        #expect(tracker.armed)
    }

    @Test func aShortPullCancelsAFlickCommitsAndAPullTakenBackCancels() {
        let short = swipe([(14, 0), (50, 0)])
        #expect(short.pull == 36)
        #expect(short.outcome(velocityX: 100) == .cancel)
        #expect(short.outcome(velocityX: 700) == .commit(.leading))
        let far = swipe([(14, 0), (110, 0)])
        #expect(far.outcome(velocityX: 0) == .commit(.leading))
        #expect(far.outcome(velocityX: -500) == .cancel)
    }

    @Test func aRowWithNoUndoNeverCommitsOnAFlick() {
        let short = swipe([(14, 0), (50, 0)])
        #expect(short.outcome(velocityX: 900) == .commit(.leading))
        #expect(short.outcome(velocityX: 900, allowsFlick: false) == .cancel)
        let far = swipe([(14, 0), (110, 0)])
        #expect(far.outcome(velocityX: 0, allowsFlick: false) == .commit(.leading))
        #expect(far.outcome(velocityX: -500, allowsFlick: false) == .cancel)
    }

    @Test func aLeftPullCommitsToTheTrailingSide() {
        let tracker = swipe([(-14, 0), (-70, 0), (-110, 0)])
        #expect(tracker.pull == -96)
        #expect(tracker.outcome(velocityX: 0) == .commit(.trailing))
        #expect(tracker.outcome(velocityX: 500) == .cancel)
    }

    @Test func thePullIsHeldInsideTheLimit() {
        let tracker = swipe([(14, 0), (400, 0)], limit: 120)
        #expect(tracker.pull == 120)
    }

    // MARK: Pages and pulls

    @Test func aPageTurnNeedsDistanceOrAFlickAndAClearSidewaysLead() {
        // Left to the next page, right to the previous.
        #expect(GestureTuning.pageTurn(startX: 200, dx: -90, dy: 10, velocityX: -100) == 1)
        #expect(GestureTuning.pageTurn(startX: 200, dx: 90, dy: 10, velocityX: 100) == -1)
        // Short and slow, or a twitch, does nothing.
        #expect(GestureTuning.pageTurn(startX: 200, dx: -40, dy: 5, velocityX: -200) == 0)
        #expect(GestureTuning.pageTurn(startX: 200, dx: -16, dy: 2, velocityX: -900) == 0)
        // Short but a real flick turns.
        #expect(GestureTuning.pageTurn(startX: 200, dx: -40, dy: 5, velocityX: -700) == 1)
        // A diagonal (leads by less than 2x) is a scroll.
        #expect(GestureTuning.pageTurn(startX: 200, dx: -90, dy: 60, velocityX: -300) == 0)
        // Taken back at lift, it cancels.
        #expect(GestureTuning.pageTurn(startX: 200, dx: -90, dy: 4, velocityX: 400) == 0)
    }

    @Test func aPageTurnToThePreviousPageCannotStartInTheBackEdge() {
        #expect(GestureTuning.pageTurn(startX: 10, dx: 120, dy: 4, velocityX: 400) == 0)
        #expect(GestureTuning.pageTurn(startX: 10, dx: -120, dy: 4, velocityX: -400) == 1)
    }

    @Test func aPullNeedsDistanceOrAFlickAndAClearVerticalLead() {
        #expect(GestureTuning.pull(dx: 4, dy: 70, velocityY: 0) == 1)
        #expect(GestureTuning.pull(dx: 4, dy: -70, velocityY: 0) == -1)
        #expect(GestureTuning.pull(dx: 4, dy: 30, velocityY: 200) == 0)
        #expect(GestureTuning.pull(dx: 4, dy: 34, velocityY: 800) == 1)
        #expect(GestureTuning.pull(dx: 40, dy: 70, velocityY: 0) == 0)
        #expect(GestureTuning.pull(dx: 4, dy: 70, velocityY: -600) == 0)
    }

    // MARK: Action words and stacks

    @Test func anActionWordLeansThenRollsThenCommitsOnlyWhenRolled() {
        #expect(GestureTuning.actionWordStage(travel: 0) == .rest)
        #expect(GestureTuning.actionWordStage(travel: -30) == .rest)
        #expect(GestureTuning.actionWordStage(travel: 17.9) == .rest)
        #expect(GestureTuning.actionWordStage(travel: 18) == .lean)
        #expect(GestureTuning.actionWordStage(travel: 55.9) == .lean)
        #expect(GestureTuning.actionWordStage(travel: 56) == .rolled)
        #expect(!GestureTuning.actionWordCommits(travel: 40, velocity: 900))
        #expect(GestureTuning.actionWordCommits(travel: 60, velocity: 0))
        #expect(!GestureTuning.actionWordCommits(travel: 60, velocity: -400))
    }

    @Test func aStackSwipeCommitsPastThirtyFivePercentOrOnAFlick() {
        #expect(GestureTuning.stackCommits(travel: 120, width: 300, velocity: 0))
        #expect(!GestureTuning.stackCommits(travel: 90, width: 300, velocity: 100))
        #expect(GestureTuning.stackCommits(travel: 90, width: 300, velocity: 700))
        #expect(!GestureTuning.stackCommits(travel: 120, width: 300, velocity: -500))
        #expect(!GestureTuning.stackCommits(travel: 120, width: 0, velocity: 0))
    }

    // MARK: Hold-off

    @Test func aTapRightAfterAScrollerLetsGoDoesNotCount() {
        var holdoff = TapHoldoff()
        #expect(holdoff.allowsTap(at: 1))
        holdoff.moved(at: 10)
        #expect(!holdoff.allowsTap(at: 10.05))
        #expect(holdoff.allowsTap(at: 10.2))
    }
}
